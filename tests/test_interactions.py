"""Milestone 1 tests: PING/PONG and signature verification, run fully offline."""

import json

import config
import main
import pytest
from fastapi.testclient import TestClient
from nacl.signing import SigningKey


@pytest.fixture
def signing_key():
    return SigningKey.generate()


@pytest.fixture
def client(signing_key, monkeypatch):
    monkeypatch.setattr(config, "DISCORD_PUBLIC_KEY", signing_key.verify_key.encode().hex())
    return TestClient(main.app)


def _signed_headers(signing_key, body, timestamp="1700000000"):
    signed = signing_key.sign(timestamp.encode() + body)
    return {
        "X-Signature-Ed25519": signed.signature.hex(),
        "X-Signature-Timestamp": timestamp,
        "Content-Type": "application/json",
    }


def test_ping_returns_pong(client, signing_key):
    body = json.dumps({"type": 1}).encode()
    resp = client.post("/interactions", content=body, headers=_signed_headers(signing_key, body))
    assert resp.status_code == 200
    assert resp.json() == {"type": 1}


def test_bad_signature_is_rejected(client, signing_key):
    body = json.dumps({"type": 1}).encode()
    headers = _signed_headers(signing_key, body)
    headers["X-Signature-Ed25519"] = "00" * 64
    resp = client.post("/interactions", content=body, headers=headers)
    assert resp.status_code == 401


def test_missing_signature_is_rejected(client):
    body = json.dumps({"type": 1}).encode()
    resp = client.post("/interactions", content=body, headers={"Content-Type": "application/json"})
    assert resp.status_code == 401


def test_application_command_defers(client, signing_key):
    body = json.dumps({"type": 2, "id": "1", "token": "abc", "data": {"name": "ask"}}).encode()
    resp = client.post("/interactions", content=body, headers=_signed_headers(signing_key, body))
    assert resp.status_code == 200
    assert resp.json() == {"type": 5}
