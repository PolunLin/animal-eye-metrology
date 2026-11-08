"""Local CPU HTTP adapter for the existing, unverified-candidate inference pipeline."""

import logging
import os
import re
from pathlib import Path
from threading import Lock
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from animal_eye_metrology.core.execution_lock import InferenceBusyError
from animal_eye_metrology.inference import inference_pipeline as pipeline
from animal_eye_metrology.paths import PROJECT_ROOT

LOGGER = logging.getLogger(__name__)
ROOT = PROJECT_ROOT
RUN_ID = re.compile(r"\d{8}T\d{6}Z_[0-9a-f]{12}\Z")


class ComponentStatus(BaseModel):
    """Readiness is local file availability, not a guarantee of model accuracy."""

    component: str
    status: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    ready: bool
    authentication: str = "none_local_demo"
    checks: list[ComponentStatus]


def create_app(config_path: Path | None = None) -> FastAPI:
    """Build an app without loading weights; each API process accepts one inference at a time."""
    app = FastAPI(
        title="Animal Eye Metrology API",
        version="0.1.0",
        description=(
            "Single-image CPU research demonstration. Distances are unverified mask-centroid "
            "distances in pixels, not pupil-center or physical distances. No login is implemented."
        ),
    )
    app.state.config_path = Path(
        config_path
        or os.environ.get("AEM_INFERENCE_CONFIG", ROOT / "configs/inference.runtime.toml")
    ).resolve()
    app.state.inference_lock = Lock()
    app.state.runner = pipeline.run_inference

    def config() -> dict[str, Any]:
        try:
            return pipeline.load_config(app.state.config_path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            LOGGER.warning("API configuration unavailable: %s", error)
            raise HTTPException(
                503, "Inference configuration is unavailable or invalid."
            ) from error

    def readiness(cfg: dict[str, Any]) -> list[dict[str, str]]:
        try:
            return pipeline.preflight(cfg)
        except (OSError, ValueError, KeyError, TypeError) as error:
            LOGGER.warning("API readiness check failed: %s", error)
            raise HTTPException(503, "Inference dependencies could not be verified.") from error

    def run_directory(run_id: str) -> Path:
        if not RUN_ID.fullmatch(run_id):
            raise HTTPException(404, "Run not found.")
        root = config()["output_directory"].resolve()
        directory = (root / run_id).resolve()
        if directory.parent != root or not directory.is_dir():
            raise HTTPException(404, "Run not found.")
        return directory

    def result_in(directory: Path) -> dict[str, Any]:
        result_path = directory / "result.json"
        if result_path.is_symlink() or not result_path.is_file():
            raise HTTPException(404, "Run result not found.")
        try:
            result = pipeline.load_result(directory)
            if not isinstance(result, dict):
                raise ValueError("Result must be a JSON object.")
            return result
        except (OSError, ValueError) as error:
            LOGGER.exception("Cannot read run result: %s", directory.name)
            raise HTTPException(500, "Saved run result could not be read.") from error

    @app.get("/health", response_model=HealthResponse, tags=["Service"])
    def health() -> HealthResponse:
        """Check local configuration, interpreters, sources and checkpoint hashes; no inference."""
        checks = readiness(config())
        ready = all(item["status"] == "ready" for item in checks)
        response = HealthResponse(
            status="ready" if ready else "not_ready", ready=ready, checks=checks
        )
        if not ready:
            raise HTTPException(503, response.model_dump())
        return response

    @app.post(
        "/predict",
        tags=["Inference"],
        responses={
            400: {"description": "Invalid image content."},
            409: {"description": "Another inference is running for the shared output directory."},
            413: {"description": "Image exceeds the configured byte limit."},
            503: {"description": "Local inference dependencies are unavailable."},
        },
    )
    def predict(
        file: Annotated[UploadFile, File(description="One actual JPEG or PNG image.")],
    ) -> dict[str, Any]:
        """Run the existing pipeline synchronously; inspect result.status for model failures."""
        cfg = config()
        limit = cfg["max_upload_bytes"]
        try:
            data = file.file.read(limit + 1)
        finally:
            file.file.close()
        if len(data) > limit:
            raise HTTPException(413, "Image exceeds the configured byte limit.")
        try:
            # Check content, dimensions and animation before any worker or output is created.
            pipeline.normalize_image(data, cfg)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if not app.state.inference_lock.acquire(blocking=False):
            raise HTTPException(409, "Another inference is running; retry after it finishes.")
        try:
            checks = readiness(cfg)
            if any(item["status"] != "ready" for item in checks):
                raise HTTPException(
                    503, {"message": "Local dependencies are not ready.", "checks": checks}
                )
            directory = app.state.runner(
                data, file.filename or "upload", app.state.config_path, ("A_r100",)
            )
            return result_in(directory)
        except InferenceBusyError as error:
            raise HTTPException(409, str(error)) from error
        except HTTPException:
            raise
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
            LOGGER.exception("API inference failed before a readable result was returned.")
            raise HTTPException(500, "Inference failed; inspect the server logs.") from error
        finally:
            app.state.inference_lock.release()

    @app.get("/runs/{run_id}", tags=["Results"])
    def get_run(run_id: str) -> dict[str, Any]:
        """Read a saved result without loading models or starting another inference."""
        return result_in(run_directory(run_id))

    @app.get("/runs/{run_id}/artifacts/{artifact_key}", tags=["Results"])
    def get_artifact(run_id: str, artifact_key: str) -> FileResponse:
        """Download only a relative file registered in this run's artifacts mapping."""
        directory = run_directory(run_id)
        result = result_in(directory)
        relative = result.get("artifacts", {}).get(artifact_key)
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise HTTPException(404, "Artifact not found.")
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise HTTPException(404, "Artifact not found.")
        return FileResponse(path, filename=path.name)

    return app


app = create_app()
