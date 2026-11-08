"""All fixtures are invented geometry for testing, never COCO observations."""

import os
import socket
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def isolate_application_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fixtures select their own workspace/config, independent of Docker or host defaults."""
    for key in tuple(os.environ):
        if key.startswith("AEM_"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def deny_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Tests must not access the network or download data.")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
