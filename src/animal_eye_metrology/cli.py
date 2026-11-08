"""Offline command-line entry points with structured logs."""

import argparse
import json
import logging
import sys
import tomllib
from dataclasses import replace
from pathlib import Path
from typing import Any

from animal_eye_metrology.core.config import load_config
from animal_eye_metrology.core.errors import Phase1Error
from animal_eye_metrology.core.verification import verify_file
from animal_eye_metrology.data.analysis import analyze
from animal_eye_metrology.data.coco import load_coco
from animal_eye_metrology.data.outputs import validate_output, write_outputs

LOGGER = logging.getLogger("animal_eye_metrology")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "level": record.levelname,
                "event": record.getMessage(),
                **getattr(record, "details", {}),
            },
            ensure_ascii=False,
        )


def _log(event: str, **details: Any) -> None:
    LOGGER.info(event, extra={"details": details})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline COCO Phase 1 analysis; never downloads data."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    analysis = commands.add_parser("analyze", help="Analyze local COCO instance annotations.")
    analysis.add_argument("--config", type=Path, required=True)
    analysis.add_argument(
        "--output", type=Path, help="Override output; relative to the working directory."
    )
    contact = analysis.add_mutually_exclusive_group()
    contact.add_argument("--contact-sheets", dest="contact_sheets", action="store_true")
    contact.add_argument("--no-contact-sheets", dest="contact_sheets", action="store_false")
    analysis.set_defaults(contact_sheets=None)
    verify = commands.add_parser(
        "verify-file", help="Hash a local file and optionally check ZIP CRCs."
    )
    verify.add_argument("path", type=Path)
    verify.add_argument("--expected-sha256", default="")
    verify.add_argument("--zip", dest="check_zip", action="store_true")
    verify.add_argument("--required-member", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    options = _parser().parse_args(arguments)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    LOGGER.handlers = [handler]
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    try:
        if options.command == "verify-file":
            result = verify_file(
                options.path,
                expected_sha256=options.expected_sha256,
                check_zip=options.check_zip,
                required_members=tuple(options.required_member),
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        config = load_config(options.config)
        if options.output is not None:
            config = replace(config, output=options.output.expanduser().resolve())
        if options.contact_sheets is not None:
            config = replace(
                config,
                contact_sheets=replace(config.contact_sheets, enabled=options.contact_sheets),
            )
        validate_output(config)
        _log("analysis_started", annotations=str(config.annotations), output=str(config.output))
        dataset = load_coco(
            config.annotations,
            expected_animal_categories=config.expected_animal_categories,
            expected_sha256=config.expected_annotations_sha256,
        )
        result = analyze(dataset)
        manifest = write_outputs(dataset, result, config, arguments)
        _log("analysis_completed", output=str(config.output), summary=manifest["summary"])
        return 0
    except (Phase1Error, OSError, tomllib.TOMLDecodeError, UnicodeError) as error:
        LOGGER.error("analysis_failed", extra={"details": {"error": str(error)}})
        return 2
