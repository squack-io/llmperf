from typing import List
from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.ray_clients.relace_client import RelaceClient
from llmperf.ray_clients.relace_reranker_client import RelaceRerankerClient

SUPPORTED_APIS = ["openai", "relace", "relace-reranker"]


def construct_clients(llm_api: str, num_clients: int):
    """Construct LLMClients that will be used to make requests to the LLM API.

    Args:
        llm_api: The name of the LLM API to use.
        num_clients: The number of concurrent requests to make.

    Returns:
        The constructed LLMCLients

    """
    if llm_api == "openai":
        clients = [OpenAIChatCompletionsClient.remote() for _ in range(num_clients)]
    elif llm_api == "relace":
        clients = [RelaceClient.remote() for _ in range(num_clients)]
    elif llm_api == "relace-reranker":
        clients = [RelaceRerankerClient.remote() for _ in range(num_clients)]
    else:
        raise ValueError(
            f"llm_api must be one of the supported LLM APIs: {SUPPORTED_APIS}"
        )

    return clients
