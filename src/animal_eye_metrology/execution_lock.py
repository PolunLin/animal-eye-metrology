"""One advisory CPU execution lock shared by local API, UI and CLI processes."""

import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class InferenceBusyError(RuntimeError):
    """Another process already holds the inference lock for this output directory."""


@contextmanager
def inference_execution_lock(output_directory: Path) -> Iterator[None]:
    """Reject concurrent execution without queueing; Linux/macOS local filesystems only.

    The lock file persists: unlinking it would let another caller create and lock
    a different inode while an older caller still holds the original lock.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / ".inference.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise InferenceBusyError(
                "Another inference is running for this output directory; retry after it finishes."
            ) from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
