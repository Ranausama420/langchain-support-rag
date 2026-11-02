import pytest
from langchain_core.messages import AIMessage

from support_rag.config import ModelSpec
from support_rag.models import generate, make_client
from support_rag.schemas import Evidence


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_missing_credentials(settings, provider):
    settings.openai_api_key = ""
    settings.azure_openai_api_key = ""
    with pytest.raises(ValueError):
        make_client(ModelSpec(alias="test", provider=provider, model="test-model"), settings)


def test_live_response_parsing_with_fake_transport(settings, monkeypatch):
    class Client:
        def invoke(self, messages):
            assert "untrusted" in messages[0].content
            return AIMessage(
                content='{"insufficient_evidence":false,"claims":'
                '[{"text":"Token expired","citation_ids":["a"]}]}',
                usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            )

    monkeypatch.setattr("support_rag.models.make_client", lambda *args: Client())
    draft, usage = generate(
        ModelSpec(alias="test", provider="openai", model="test-model"),
        settings,
        "AUTH-104?",
        [Evidence(id="a", text="Token expired", source="auth.md", section="AUTH-104", similarity=0.8)],
    )
    assert draft.claims[0].citation_ids == ["a"] and usage["total_tokens"] == 30


def test_invalid_json_response_fails(settings, monkeypatch):
    class Client:
        def invoke(self, messages):
            return AIMessage(content="not JSON")

    monkeypatch.setattr("support_rag.models.make_client", lambda *args: Client())
    with pytest.raises(ValueError):
        generate(ModelSpec(alias="test", provider="openai", model="test-model"), settings, "test", [])
