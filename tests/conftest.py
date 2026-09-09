import socket

import pytest


@pytest.fixture(autouse=True)
def controlled_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_MODEL", raising=False)
    monkeypatch.delenv("APPLICANT_EMAIL", raising=False)
    monkeypatch.setenv("FOOD_ORDER_LOG_PATH", str(tmp_path / "default-turns.jsonl"))

    def forbid_network(*args, **kwargs):
        raise AssertionError("Default tests must use controlled external dependencies")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
