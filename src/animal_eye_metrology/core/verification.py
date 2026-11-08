"""Local integrity checks, without extraction or downloading."""

import hashlib
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from animal_eye_metrology.core.errors import Phase1Error


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_file(
    path: Path,
    *,
    expected_sha256: str = "",
    check_zip: bool = False,
    required_members: tuple[str, ...] = (),
) -> dict[str, Any]:
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256.lower():
        raise Phase1Error(f"SHA-256 mismatch for {path}.")
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "reference_digest_matched": True if expected_sha256 else None,
        "authenticity_verified": False,
        "note": "A local hash or ZIP CRC is not proof of official provenance.",
    }
    if required_members and not check_zip:
        raise Phase1Error("Required ZIP members need --zip.")
    if check_zip:
        try:
            with ZipFile(path) as archive:
                missing = set(required_members) - set(archive.namelist())
                if missing:
                    raise Phase1Error(f"ZIP is missing required members: {sorted(missing)}")
                failed = archive.testzip()
                if failed:
                    raise Phase1Error(f"ZIP CRC check failed for member {failed}.")
                record["zip"] = {
                    "crc_passed": True,
                    "member_count": len(archive.infolist()),
                    "required_members": list(required_members),
                }
        except (BadZipFile, RuntimeError, NotImplementedError) as error:
            raise Phase1Error(f"Unable to verify ZIP: {error}") from error
    return record
