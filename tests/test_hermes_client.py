from __future__ import annotations

import httpx
import pytest

from hermes_mcp.hermes_client import HermesClient, HermesError


def _client(timeout: int = 60) -> HermesClient:
    return HermesClient(
        api_url="http://127.0.0.1:8642",
        api_key="k" * 32,
        model="hermes-agent",
        timeout_seconds=timeout,
    )


def _ok_response(content: str = "the answer") -> dict[str, object]:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "model": "hermes",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def test_constructor_validates_required_args() -> None:
    with pytest.raises(ValueError, match="api_url is required"):
        HermesClient(api_url="", api_key="k", model="m", timeout_seconds=10)
    with pytest.raises(ValueError, match="api_key is required"):
        HermesClient(api_url="http://localhost", api_key="", model="m", timeout_seconds=10)


def test_ask_posts_chat_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return httpx.Response(200, json=_ok_response("hi from hermes"))

    monkeypatch.setattr("hermes_mcp.hermes_client.httpx.post", fake_post)
    out = _client().ask("ping")
    assert out == "hi from hermes"
    assert captured["url"] == "http://127.0.0.1:8642/v1/chat/completions"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["json"] == {  # type: ignore[index]
        "model": "hermes-agent",
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
    }
    assert kwargs["headers"]["Authorization"] == "Bearer " + "k" * 32  # type: ignore[index]
    assert "X-Hermes-Session-Id" not in kwargs["headers"]  # type: ignore[index]


def test_ask_forwards_session_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured["kwargs"] = kwargs
        return httpx.Response(200, json=_ok_response())

    monkeypatch.setattr("hermes_mcp.hermes_client.httpx.post", fake_post)
    _client().ask("ping", session_id="sess-1")
    headers = captured["kwargs"]["headers"]  # type: ignore[index]
    assert headers["X-Hermes-Session-Id"] == "sess-1"


def test_ask_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_mcp.hermes_client.httpx.post",
        lambda *a, **kw: httpx.Response(200, json=_ok_response("  spaced out  \n")),
    )
    assert _client().ask("ping") == "spaced out"


def test_ask_translates_401_to_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_mcp.hermes_client.httpx.post",
        lambda *a, **kw: httpx.Response(401, text="unauthorized"),
    )
    with pytest.raises(HermesError, match="rejected the API key"):
        _client().ask("ping")


def test_ask_propagates_non_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_mcp.hermes_client.httpx.post",
        lambda *a, **kw: httpx.Response(500, text="boom"),
    )
    with pytest.raises(HermesError, match="returned HTTP 500"):
        _client().ask("ping")


def test_ask_translates_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_kw: object) -> httpx.Response:
        raise httpx.ConnectTimeout("slow")

    monkeypatch.setattr("hermes_mcp.hermes_client.httpx.post", boom)
    with pytest.raises(HermesError, match="timed out after"):
        _client().ask("ping")


def test_ask_translates_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_kw: object) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("hermes_mcp.hermes_client.httpx.post", boom)
    with pytest.raises(HermesError, match="request failed"):
        _client().ask("ping")


def test_ask_handles_malformed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "hermes_mcp.hermes_client.httpx.post",
        lambda *a, **kw: httpx.Response(200, json={"choices": []}),
    )
    with pytest.raises(HermesError, match="malformed"):
        _client().ask("ping")


def test_ask_rejects_non_string_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """Some OpenAI-compatible servers return content as a list of segments;
    we don't try to flatten — fail fast so the operator notices."""

    def fake_post(*_a: object, **_kw: object) -> httpx.Response:
        body = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
                    "finish_reason": "stop",
                }
            ]
        }
        return httpx.Response(200, json=body)

    monkeypatch.setattr("hermes_mcp.hermes_client.httpx.post", fake_post)
    with pytest.raises(HermesError, match="content was list"):
        _client().ask("ping")


def test_ask_does_not_echo_response_body_in_user_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misbehaving gateway must not be able to inject content into the
    user-visible error string. Body lands in DEBUG only."""
    secret_marker = "GATEWAY_RESPONSE_BODY_SHOULD_NOT_LEAK"
    monkeypatch.setattr(
        "hermes_mcp.hermes_client.httpx.post",
        lambda *a, **kw: httpx.Response(500, text=secret_marker),
    )
    try:
        _client().ask("ping")
    except HermesError as exc:
        assert secret_marker not in str(exc)
    else:
        raise AssertionError("expected HermesError")


def test_ask_does_not_log_prompt_at_info(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Privacy invariant: prompt body never appears in INFO logs."""
    monkeypatch.setattr(
        "hermes_mcp.hermes_client.httpx.post",
        lambda *a, **kw: httpx.Response(200, json=_ok_response()),
    )
    with caplog.at_level("INFO", logger="hermes_mcp.hermes_client"):
        _client().ask("VERY-PRIVATE-PROMPT-CONTENT-XYZ")
    for rec in caplog.records:
        assert "VERY-PRIVATE-PROMPT-CONTENT-XYZ" not in rec.message


# --- profile routing ------------------------------------------------------------


def _profile_client() -> HermesClient:
    return HermesClient(
        api_url="http://127.0.0.1:8642",
        api_key="k" * 32,
        model="hermes-agent",
        timeout_seconds=60,
        profile_routes={
            "mgr-plattform": "http://127.0.0.1:8642/p/mgr-plattform",
            "planer": "http://127.0.0.1:8642/p/planer",
        },
        profile_keys={"mgr-plattform": "p" * 32},
    )


def test_ask_routes_profile_to_profile_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return httpx.Response(200, json=_ok_response("ok"))

    monkeypatch.setattr("hermes_mcp.hermes_client.httpx.post", fake_post)
    out = _profile_client().ask("ping", profile="mgr-plattform")
    assert out == "ok"
    assert captured["url"] == "http://127.0.0.1:8642/p/mgr-plattform/v1/chat/completions"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["headers"]["Authorization"] == "Bearer " + "p" * 32  # type: ignore[index]


def test_ask_profile_falls_back_to_base_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """planer has a route but no HERMES_API_KEY_PLANER -> base key is used."""

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, json=_ok_response("ok"))

    monkeypatch.setattr("hermes_mcp.hermes_client.httpx.post", fake_post)
    client = HermesClient(
        api_url="http://127.0.0.1:8642",
        api_key="k" * 32,
        model="hermes-agent",
        timeout_seconds=60,
        profile_routes={"planer": "http://127.0.0.1:8642/p/planer"},
    )
    assert client.ask("ping", profile="planer") == "ok"


def test_ask_profile_default_and_none_hit_base_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urls: list[str] = []

    def fake_post(url: str, **kwargs: object) -> httpx.Response:
        urls.append(url)
        return httpx.Response(200, json=_ok_response("ok"))

    monkeypatch.setattr("hermes_mcp.hermes_client.httpx.post", fake_post)
    client = _profile_client()
    client.ask("ping")
    client.ask("ping", profile=None)
    client.ask("ping", profile="default")
    assert urls == ["http://127.0.0.1:8642/v1/chat/completions"] * 3


def test_ask_unknown_profile_rejected_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The allowlist is the boundary: unknown names fail fast and must not
    silently fall back to the base endpoint."""

    def fail_post(*args: object, **kwargs: object) -> httpx.Response:
        raise AssertionError("httpx.post must not be called for unknown profiles")

    monkeypatch.setattr("hermes_mcp.hermes_client.httpx.post", fail_post)
    with pytest.raises(HermesError, match="unknown profile 'mgr-voice'"):
        _profile_client().ask("ping", profile="mgr-voice")


def test_ask_profile_without_routes_rejects_everything_but_default() -> None:
    """A bridge configured without HERMES_PROFILE_ROUTES stays default-only:
    any profile name is rejected (fail closed)."""
    with pytest.raises(HermesError, match="unknown profile 'planer'"):
        _client().ask("ping", profile="planer")
