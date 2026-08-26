"""FPL client behaviour around caching and failure."""

import json

import httpx
import pytest

from fpl.data.fpl_client import FPLClient, FPLNotFound, FPLUnavailable


def test_offline_uses_cache(tmp_path):
    (tmp_path / "bootstrap-static.json").write_text(json.dumps({"elements": []}))
    client = FPLClient(cache_dir=tmp_path, offline=True)
    assert client.bootstrap() == {"elements": []}


def test_offline_without_cache_fails_clearly(tmp_path):
    client = FPLClient(cache_dir=tmp_path, offline=True)
    with pytest.raises(FPLUnavailable, match="offline"):
        client.bootstrap()


def test_network_failure_falls_back_to_stale_cache(tmp_path, monkeypatch):
    """A flaky connection should degrade to old data, not to an error."""
    (tmp_path / "fixtures.json").write_text(json.dumps([{"id": 1}]))

    def explode(*args, **kwargs):
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(httpx, "get", explode)
    client = FPLClient(cache_dir=tmp_path, ttl_seconds=0)
    assert client.fixtures() == [{"id": 1}]


def test_missing_entry_raises_not_found_even_with_a_cached_copy(tmp_path, monkeypatch):
    """A 404 must never be masked by stale data — it means the id is wrong."""
    (tmp_path / "entry__1807966.json").write_text(json.dumps({"id": 1807966}))

    def not_found(*args, **kwargs):
        return httpx.Response(404, request=httpx.Request("GET", "http://x/"))

    monkeypatch.setattr(httpx, "get", not_found)
    client = FPLClient(cache_dir=tmp_path, ttl_seconds=0)
    with pytest.raises(FPLNotFound):
        client.entry(1807966)


def test_not_found_is_a_kind_of_unavailable():
    assert issubclass(FPLNotFound, FPLUnavailable)


def test_successful_fetch_is_cached(tmp_path, monkeypatch):
    payload = {"events": []}
    calls = []

    def ok(*args, **kwargs):
        calls.append(args)
        return httpx.Response(
            200, json=payload, request=httpx.Request("GET", "http://x/")
        )

    monkeypatch.setattr(httpx, "get", ok)
    client = FPLClient(cache_dir=tmp_path, ttl_seconds=3600)
    assert client.bootstrap() == payload
    assert client.bootstrap() == payload
    assert len(calls) == 1, "second call should have been served from cache"
