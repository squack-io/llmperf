from typing import List
from llmperf.ray_clients.litellm_client import LiteLLMClient
from llmperf.ray_clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from llmperf.ray_clients.sagemaker_client import SageMakerClient
from llmperf.ray_clients.vertexai_client import VertexAIClient
from llmperf.ray_llm_client import LLMClient
from llmperf.ray_clients.fast_apply_client import FastApplyClient
from llmperf.ray_clients.anthropic_plandex_client import AnthropicPlandexClient
from llmperf.ray_clients.plandex_client import PlandexClient

SUPPORTED_APIS = ["openai", "anthropic_plandex", "litellm", "fast_apply", "plandex"]


def construct_clients(llm_api: str, num_clients: int) -> List[LLMClient]:
    """Construct LLMClients that will be used to make requests to the LLM API.

    Args:
        llm_api: The name of the LLM API to use.
        num_clients: The number of concurrent requests to make.

    Returns:
        The constructed LLMCLients

    """
    if llm_api == "openai":
        clients = [OpenAIChatCompletionsClient.remote() for _ in range(num_clients)]
    elif llm_api == "fast_apply":
        clients = [FastApplyClient.remote() for _ in range(num_clients)]
    elif llm_api == "anthropic_plandex":
        clients = [AnthropicPlandexClient.remote() for _ in range(num_clients)]
    elif llm_api == "plandex":
        clients = [PlandexClient.remote() for _ in range(num_clients)]
    elif llm_api == "sagemaker":
        clients = [SageMakerClient.remote() for _ in range(num_clients)]
    elif llm_api == "vertexai":
        clients = [VertexAIClient.remote() for _ in range(num_clients)]
    elif llm_api in SUPPORTED_APIS:
        clients = [LiteLLMClient.remote() for _ in range(num_clients)]
    else:
        raise ValueError(
            f"llm_api must be one of the supported LLM APIs: {SUPPORTED_APIS}"
        )

    return clients
