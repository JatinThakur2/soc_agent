"""LLM factory — returns ChatAnthropic or ChatOllama based on config backend."""
import os
from dotenv import load_dotenv


def make_llm(cfg):
    """Return a LangChain chat model configured from config.yaml agents section."""
    agent_cfg = cfg["agents"]
    backend = agent_cfg.get("backend", "anthropic")
    temperature = agent_cfg.get("temperature", 0.0)
    max_tokens = agent_cfg.get("max_tokens", 2000)

    if backend == "ollama":
        from langchain_ollama import ChatOllama
        model = agent_cfg.get("ollama_model", "phi3.5")
        return ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
        )

    load_dotenv()
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=agent_cfg["llm_model"],
        temperature=temperature,
        max_tokens=max_tokens,
    )
