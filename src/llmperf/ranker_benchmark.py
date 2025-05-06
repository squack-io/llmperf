from collections.abc import Iterable
import time
import random
from typing import Any, Dict, List, Tuple
import pandas as pd
from llmperf import common_metrics
from llmperf.common import construct_clients
from llmperf.models import RankerConfig
from llmperf.requests_launcher import RequestsLauncher
from tqdm import tqdm


def get_token_throughput_latencies(
    csv_path: str,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    random.seed(11111)
    # Load all messages from jsonl file
    rows = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        rows.append(row.to_dict())
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
        query = rows[6]["query"]
        codebase = rows[6]["codebase"]
        ranker_config = RankerConfig(
            query=query,
            codebase=codebase,
            llm_api=llm_api,
        )
        req_launcher.launch_ranker_requests(ranker_config)
        iter += 1

        if iter % num_concurrent_requests == 0:
            outs = req_launcher.get_next_ready()
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, _ = out
                all_metrics.append(request_metrics)
            completed_requests.extend(all_metrics)

        pbar.update(len(completed_requests) - num_completed_requests)
        num_completed_requests = len(completed_requests)
    pbar.close()
    end_time = time.monotonic()
    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")
    outs = req_launcher.get_next_ready()
    all_metrics = []
    for out in outs:
        request_metrics, gen_text, _ = out
        all_metrics.append(request_metrics)
    completed_requests.extend(all_metrics)
    ret = metrics_summary(completed_requests, start_time, end_time)
    metadata: Dict[str, Any] = {
        "num_concurrent_requests": num_concurrent_requests,
    }
    metadata["results"] = ret
    return metadata, completed_requests


def metrics_summary(
    metrics: List[Dict[str, Any]], start_time: float, end_time: float
) -> Dict[str, Any]:
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
        common_metrics.E2E_LAT,
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
    test_timeout_s: int,
    max_num_completed_requests: int,
    num_concurrent_requests: int,
    csv_path: str,
):
    summary, individual_responses = get_token_throughput_latencies(
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        num_concurrent_requests=num_concurrent_requests,
        csv_path=csv_path,
    )
    return summary, individual_responses
