from collections.abc import Iterable
import time
import random
import json
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from llmperf import common_metrics
from llmperf.common import construct_clients
from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


def get_token_throughput_latencies(
    model: str,
    tokenizer: PreTrainedTokenizerFast,
    jsonl_path: str,
    max_output_tokens: int,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.
    Args:
        model: The name of the model to query.
        tokenizer: The tokenizer to use for counting tokens.
        jsonl_path: Path to jsonl file containing messages in format {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]}
        max_output_tokens: The maximum number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".
    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    random.seed(11111)
    # Load all messages from jsonl file
    messages = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            messages.append(data)
    get_token_length = lambda text: len(tokenizer.encode(text))
    if not additional_sampling_params:
        additional_sampling_params = {}
    clients = construct_clients(llm_api=llm_api, num_clients=num_concurrent_requests)
    req_launcher = RequestsLauncher(clients)
    completed_requests = []
    num_completed_requests = 0
    start_time = time.monotonic()
    iter = 0
    pbar = tqdm(total=max_num_completed_requests)
    while (
        time.monotonic() - start_time < test_timeout_s
        and len(completed_requests) < max_num_completed_requests
    ):
        iter += 1
        default_sampling_params = {"max_tokens": max_output_tokens}
        default_sampling_params.update(additional_sampling_params)
        # Randomly select a message from the jsonl and extract user content
        message = random.choice(messages)
        system_prompt = message["messages"][0]["content"]
        user_prompt = message["messages"][1]["content"]
        num_input_tokens = get_token_length(system_prompt + user_prompt)
        request_config = RequestConfig(
            model=model,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            num_input_tokens=num_input_tokens,
            sampling_params=default_sampling_params,
            llm_api=llm_api,
        )
        req_launcher.launch_requests(request_config)
        if not (iter % num_concurrent_requests):
            outs = req_launcher.get_next_ready()
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, _ = out
                num_output_tokens = get_token_length(gen_text)
                if num_output_tokens:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
                else:
                    request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
                request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
                request_metrics[common_metrics.NUM_TOTAL_TOKENS] = (
                    request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
                )
                # Calculate throughput as output tokens / (total time - TTFT)
                generation_time = (
                    request_metrics[common_metrics.E2E_LAT]
                    - request_metrics[common_metrics.TTFT]
                )
                request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
                    num_output_tokens / generation_time if generation_time > 0 else 0
                )
                all_metrics.append(request_metrics)
            completed_requests.extend(all_metrics)
        pbar.update(len(completed_requests) - num_completed_requests)
        num_completed_requests = len(completed_requests)
    pbar.close()
    end_time = time.monotonic()
    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")
    # check one last time that there are no remaining results to collect.
    outs = req_launcher.get_next_ready()
    all_metrics = []
    for out in outs:
        request_metrics, gen_text, _ = out
        num_output_tokens = get_token_length(gen_text)
        if num_output_tokens:
            request_metrics[common_metrics.INTER_TOKEN_LAT] /= num_output_tokens
        else:
            request_metrics[common_metrics.INTER_TOKEN_LAT] = 0
        request_metrics[common_metrics.NUM_OUTPUT_TOKENS] = num_output_tokens
        request_metrics[common_metrics.NUM_TOTAL_TOKENS] = (
            request_metrics[common_metrics.NUM_INPUT_TOKENS] + num_output_tokens
        )
        # Calculate throughput as output tokens / (total time - TTFT)
        generation_time = (
            request_metrics[common_metrics.E2E_LAT]
            - request_metrics[common_metrics.TTFT]
        )
        request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
            num_output_tokens / generation_time if generation_time > 0 else 0
        )
        all_metrics.append(request_metrics)
    completed_requests.extend(all_metrics)
    print(f"Results for token benchmark for {model} queried with the {llm_api} api.\n")
    ret = metrics_summary(completed_requests, start_time, end_time)
    metadata = {
        "model": model,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
    }
    metadata["results"] = ret
    return metadata, completed_requests


def metrics_summary(
    metrics: List[Dict[str, Any]], start_time: float, end_time: float
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.
    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.
    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}

    def flatten(item):
        for sub_item in item:
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]
    for key in [
        common_metrics.INTER_TOKEN_LAT,
        common_metrics.TTFT,
        common_metrics.E2E_LAT,
        common_metrics.REQ_OUTPUT_THROUGHPUT,
        common_metrics.NUM_INPUT_TOKENS,
        common_metrics.NUM_OUTPUT_TOKENS,
    ]:
        print(key)
        ret[key] = {}
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            print(f"    {reformatted_key} = {value}")
            quantiles_reformatted_keys[reformatted_key] = value
        ret[key]["quantiles"] = quantiles_reformatted_keys
        mean = series.mean()
        print(f"    mean = {mean}")
        ret[key]["mean"] = mean
        print(f"    min = {series.min()}")
        ret[key]["min"] = series.min()
        print(f"    max = {series.max()}")
        ret[key]["max"] = series.max()
        print(f"    stddev = {series.std()}")
        ret[key]["stddev"] = series.std()
    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)
    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    print(f"Number Of Errored Requests: {num_errors}")
    error_code_frequency = dict(error_codes.value_counts())
    if num_errors:
        error_code_frequency = dict(error_codes.value_counts())
        print("Error Code Frequency")
        print(error_code_frequency)
    ret[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)
    # Calculate overall throughput as total output tokens / (total time - mean TTFT)
    mean_ttft = df_without_errored_req[common_metrics.TTFT].mean()
    generation_time = end_time - start_time - mean_ttft
    overall_output_throughput = (
        df_without_errored_req[common_metrics.NUM_OUTPUT_TOKENS].sum() / generation_time
        if generation_time > 0
        else 0
    )
    print(f"Overall Output Throughput: {overall_output_throughput}")
    ret[common_metrics.OUTPUT_THROUGHPUT] = overall_output_throughput
    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
        num_completed_requests / (end_time - start_time) * 60
    )
    print(f"Number Of Completed Requests: {num_completed_requests}")
    print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")
    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min
    return ret


def run_token_benchmark(
    llm_api: str,
    model: str,
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    max_output_tokens: int,
    tokenizer: PreTrainedTokenizerFast,
    jsonl_path: str,
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        max_output_tokens: The maximum number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
    """
    summary, individual_responses = get_token_throughput_latencies(
        model=model,
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        max_output_tokens=max_output_tokens,
        num_concurrent_requests=num_concurrent_requests,
        additional_sampling_params={},
        tokenizer=tokenizer,
        jsonl_path=jsonl_path,
    )
    return summary, individual_responses
