import unittest
from pathlib import Path

from incident_assistant.config import LLMProviderName, Settings
from incident_assistant.infrastructure.gemini_agent import GeminiAgentProvider
from incident_assistant.infrastructure.groq_agent import GroqAgentProvider
from incident_assistant.infrastructure.llm_provider_factory import create_llm_provider
from incident_assistant.infrastructure.openai_agent import OpenAIAgentProvider


class ProviderFactoryTests(unittest.TestCase):
    def settings(self, provider: LLMProviderName, **updates):
        values = {
            "database_path": Path("support.db"),
            "log_path": Path("app.log"),
            "llm_provider": provider,
            "enable_ai_agent": True,
            "gemini_api_key": "gemini-key",
            "groq_api_key": "groq-key",
            "openai_api_key": "openai-key",
        }
        values.update(updates)
        return Settings(**values)

    def test_selects_each_configured_provider(self) -> None:
        cases = (
            (LLMProviderName.GEMINI, GeminiAgentProvider),
            (LLMProviderName.GROQ, GroqAgentProvider),
            (LLMProviderName.OPENAI, OpenAIAgentProvider),
        )
        for provider_name, provider_type in cases:
            with self.subTest(provider=provider_name):
                fake_client = object()
                provider = create_llm_provider(
                    self.settings(provider_name),
                    clients={provider_name.value: fake_client},
                )
                self.assertIsInstance(provider, provider_type)
                self.assertEqual(provider.provider_name, provider_name.value)

    def test_rules_disabled_and_missing_selected_key_return_none(self) -> None:
        self.assertIsNone(create_llm_provider(self.settings(LLMProviderName.RULES)))
        self.assertIsNone(
            create_llm_provider(self.settings(LLMProviderName.GEMINI, gemini_api_key=None))
        )
        self.assertIsNone(
            create_llm_provider(self.settings(LLMProviderName.GROQ, enable_ai_agent=False))
        )

    def test_models_and_keys_are_selected_centrally(self) -> None:
        settings = self.settings(LLMProviderName.GROQ, groq_model="custom-groq")
        self.assertEqual(settings.selected_api_key, "groq-key")
        self.assertEqual(settings.selected_model, "custom-groq")
        self.assertTrue(settings.ai_provider_configured)


if __name__ == "__main__":
    unittest.main()
