from montagewright.review import _identity_review_parts


def test_review_receives_same_grounding_spec_and_reference_images(monkeypatch):
    spec, client, cache = object(), object(), object()
    calls = []

    def parts(actual, **kwargs):
        calls.append((actual, kwargs))
        return [{"type": "image", "uri": "files/approved-reference"}]

    monkeypatch.setattr("montagewright.reference_grounding.reference_prompt_parts", parts)
    result = _identity_review_parts(spec, client, cache)
    assert calls == [(spec, {"client": client, "cache": cache, "resolution": "high"})]
    assert result[-1] == {"type": "image", "uri": "files/approved-reference"}
    assert "未確認" in result[0]["text"]
    assert _identity_review_parts(None, client, cache) == []
    assert len(calls) == 1
