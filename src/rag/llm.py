"""LLM interface for generation (Groq or Ollama)."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dotenv import load_dotenv
from src.rag.config import get_config

load_dotenv()


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        pass


class GroqProvider(LLMProvider):
    """Groq API provider (free tier)."""

    def __init__(self):
        from groq import Groq
        import os
        cfg = get_config()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        self.client = Groq(api_key=api_key)
        self.model = cfg["GROQ_MODEL"]

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content


class OllamaProvider(LLMProvider):
    """Ollama local provider."""

    def __init__(self):
        import requests
        cfg = get_config()
        self.base_url = cfg["OLLAMA_BASE_URL"]
        self.model = cfg["OLLAMA_MODEL"]
        self.session = requests.Session()

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        import requests
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }
        response = self.session.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]


def get_llm_provider() -> LLMProvider:
    """Factory to get configured LLM provider."""
    cfg = get_config()
    provider = cfg["LLM_PROVIDER"].lower()
    if provider == "groq":
        return GroqProvider()
    elif provider == "ollama":
        return OllamaProvider()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


SYSTEM_PROMPT = """You are a helpful study assistant. Answer questions based ONLY on the provided context from course materials.
If the context doesn't contain the answer, say "I couldn't find that information in your course materials."
Do not include source citations in your answer text. Sources are displayed separately by the UI.
Be concise but thorough."""


def build_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """Build RAG prompt with retrieved contexts."""
    context_blocks = []
    for ctx in contexts:
        context_blocks.append(
            f"[Source: {ctx['source']}, Chunk: {ctx['chunk_id']}]\n{ctx['text']}"
        )
    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""Context from your course materials:
{context_str}

Question: {query}

Answer based on the context above:"""