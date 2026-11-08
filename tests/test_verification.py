import hashlib
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import pytest

from animal_eye_metrology.errors import Phase1Error
from animal_eye_metrology.verification import verify_file


def test_hash_and_zip_crc_and_member_check(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.zip"
    with ZipFile(path, "w", compression=ZIP_STORED) as archive:
        archive.writestr("annotations/test.json", "synthetic payload")
    before = path.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    record = verify_file(
        path, expected_sha256=digest, check_zip=True, required_members=("annotations/test.json",)
    )
    assert record["reference_digest_matched"] is True
    assert record["authenticity_verified"] is False
    assert record["zip"]["crc_passed"] is True
    assert path.read_bytes() == before
    assert not (tmp_path / "annotations").exists()
    with pytest.raises(Phase1Error, match="SHA-256 mismatch"):
        verify_file(path, expected_sha256="0" * 64)
    with pytest.raises(Phase1Error, match="missing required members"):
        verify_file(path, check_zip=True, required_members=("absent.json",))
    path.write_bytes(before.replace(b"synthetic payload", b"corrupted payload"))
    with pytest.raises(Phase1Error, match="CRC check failed"):
        verify_file(path, check_zip=True)


def test_unreferenced_hash_is_not_an_authenticity_claim(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("test")
    record = verify_file(path)
    assert record["reference_digest_matched"] is None
    assert not record["authenticity_verified"]
    with pytest.raises(Phase1Error, match="need --zip"):
        verify_file(path, required_members=("file",))
