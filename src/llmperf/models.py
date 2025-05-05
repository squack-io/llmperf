from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


class RequestConfig(BaseModel):
    """The configuration for a request to the LLM API.

    Args:
        model: The model to use.
        user_prompt: The user prompt to provide to the LLM API.
        system_prompt: The system prompt to provide to the LLM API.
        num_input_tokens: The number of tokens in the system prompt and user prompt combined.
        sampling_params: Additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions
        llm_api: The name of the LLM API to send the request to.
        metadata: Additional metadata to attach to the request for logging or validation purposes.
    """

    model: str
    user_prompt: str
    system_prompt: str
    num_input_tokens: int
    sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RankerConfig(BaseModel):
    """The configuration for a request to the LLM API.

    Args:
        model: The model to use.
        user_prompt: The user prompt to provide to the LLM API.
        system_prompt: The system prompt to provide to the LLM API.
        num_input_tokens: The number of tokens in the system prompt and user prompt combined.
        sampling_params: Additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions
        llm_api: The name of the LLM API to send the request to.
        metadata: Additional metadata to attach to the request for logging or validation purposes.
    """

    query: str
    codebase: Any
    llm_api: Optional[str] = None
    metadata: Any = None
