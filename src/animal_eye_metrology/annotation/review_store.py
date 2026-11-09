"""Atomic CSV persistence for human annotation revisions."""

import csv
import os
import tempfile
from pathlib import Path


def _write_atomic(
    path: Path,
    rows: list[dict[str, str]],
    fields: tuple[str, ...],
) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=".review-",
            suffix=".csv",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
