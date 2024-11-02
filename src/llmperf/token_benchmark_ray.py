from collections.abc import Iterable
import os
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import ray

from llmperf import common_metrics
from llmperf.common import construct_clients

from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    sample_random_positive_int,
)
from tqdm import tqdm

from transformers import PreTrainedTokenizerFast

thousand_tokens = """The Spy War: How the C.I.A. Secretly Helps Ukraine Fight Putin
For more than a decade, the United States has nurtured a secret intelligence partnership with Ukraine that is now critical for both countries in countering Russia.

Published Feb. 25, 2024Updated Feb. 28, 2024
A soldier in camouflage gear in a forest whose trees have been largely stripped of leaves.
A Ukrainian Army soldier in a forest near Russian lines this month. A C.I.A.-supported network of spy bases has been constructed in the past eight years that includes 12 secret locations along the Russian border.Tyler Hicks/The New York Times
A Ukrainian Army soldier in a forest near Russian lines this month. A C.I.A.-supported network of spy bases has been constructed in the past eight years that includes 12 secret locations along the Russian border.Tyler Hicks/The New York Times

By Adam Entous and Michael Schwirtz

Adam Entous and Michael Schwirtz conducted more than 200 interviews in Ukraine, several other European countries and the United States to report this story.

Nestled in a dense forest, the Ukrainian military base appears abandoned and destroyed, its command center a burned-out husk, a casualty of a Russian missile barrage early in the war.

But that is above ground.

Listen to this article with reporter commentary


Not far away, a discreet passageway descends to a subterranean bunker where teams of Ukrainian soldiers track Russian spy satellites and eavesdrop on conversations between Russian commanders. On one screen, a red line followed the route of an explosive drone threading through Russian air defenses from a point in central Ukraine to a target in the Russian city of Rostov.

The underground bunker, built to replace the destroyed command center in the months after Russia's invasion, is a secret nerve center of Ukraine's military.

There is also one more secret: The base is almost fully financed, and partly equipped, by the C.I.A.

"One hundred and ten percent," Gen. Serhii Dvoretskiy, a top intelligence commander, said in an interview at the base.

Now entering the third year of a war that has claimed hundreds of thousands of lives, the intelligence partnership between Washington and Kyiv is a linchpin of Ukraine's ability to defend itself. The C.I.A. and other American intelligence agencies provide intelligence for targeted missile strikes, track Russian troop movements and help support spy networks.

But the partnership is no wartime creation, nor is Ukraine the only beneficiary.

It took root a decade ago, coming together in fits and starts under three very different U.S. presidents, pushed forward by key individuals who often took daring risks. It has transformed Ukraine, whose intelligence agencies were long seen as thoroughly compromised by Russia, into one of Washington's most important intelligence partners against the Kremlin today.

A part of Malaysia Airlines Flight 17, which was shot down over Ukraine in 2014, in a field.
A part of Malaysia Airlines Flight 17, which was shot down over Ukraine in 2014, killing nearly 300 people.Mauricio Lima for The New York Times
The listening post in the Ukrainian forest is part of a C.I.A.-supported network of spy bases constructed in the past eight years that includes 12 secret locations along the Russian border. Before the war, the Ukrainians proved themselves to the Americans by collecting intercepts that helped prove Russia's involvement in the 2014 downing of a commercial jetliner, Malaysia Airlines Flight 17. The Ukrainians also helped the Americans go after the Russian operatives who meddled in the 2016 U.S. presidential election.

Around 2016, the C.I.A. began training an elite Ukrainian commando force — known as Unit 2245 — which captured Russian drones and communications gear so that C.I.A. technicians could reverse-engineer them and crack Moscow's encryption systems. (One officer in the unit was Kyrylo Budanov, now the general leading Ukraine's military intelligence.)

And the C.I.A. also helped train a new generation of Ukrainian spies who operated inside Russia, across Europe, and in Cuba and other places where the Russians have a large presence.

The relationship is so ingrained that C.I.A. officers remained at a remote location in western Ukraine when the Biden administration evacuated U.S. personnel in the weeks before Russia invaded in February 2022. During the invasion, the officers relayed critical intelligence, including where Russia was planning strikes and which weapons systems they would use.

"Without them, there would have been no way for us to resist the Russians, or to beat them," said Ivan Bakanov, who was then head of Ukraine's domestic intelligence agency, the S.B.U.

The details of this intelligence partnership, many of which are being disclosed by The New York Times for the first time, have been a closely guarded secret for a decade, discovered through lots of interviews with staff and soldiers.

"""


def get_token_throughput_latencies(
    model: str,
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    tokenizer: PreTrainedTokenizerFast,
    additional_sampling_params: Optional[Dict[str, Any]] = None,
    num_concurrent_requests: int = 1,
    max_num_completed_requests: int = 500,
    test_timeout_s=90,
    llm_api="openai",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        tokenizer: The tokenizer to use for counting tokens.
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

    get_token_length = lambda text: len(tokenizer.encode(text))

    if not additional_sampling_params:
        additional_sampling_params = {}

    clients = construct_clients(llm_api=llm_api, num_clients=num_concurrent_requests)
    req_launcher = RequestsLauncher(clients)
    completed_requests = []
    num_completed_requests = 0
    # make up prompts outside of send loop for faster benchmarking loop
    num_output_tokens_list = []
    for i in range(max_num_completed_requests):
        num_output_tokens = sample_random_positive_int(
            mean_output_tokens, stddev_output_tokens
        )
        num_output_tokens_list.append(num_output_tokens)

    start_time = time.monotonic()
    iter = 0
    pbar = tqdm(total=max_num_completed_requests)
    while (
        time.monotonic() - start_time < test_timeout_s
        and len(completed_requests) < max_num_completed_requests
    ):
        iter += 1

        default_sampling_params = {"max_tokens": num_output_tokens_list.pop()}
        default_sampling_params.update(additional_sampling_params)
        request_config = RequestConfig(
            model=model,
            prompt=((mean_input_tokens // 1000) * thousand_tokens, mean_input_tokens),
            sampling_params=default_sampling_params,
            llm_api=llm_api,
        )
        req_launcher.launch_requests(request_config)
        # Retrieving results less frequently allows for more concurrent requests
        # to be launched. This will overall reduce the amount of time it takes
        # for the test to run.
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
                request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
                    num_output_tokens / request_metrics[common_metrics.E2E_LAT]
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
        request_metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = (
            num_output_tokens / request_metrics[common_metrics.E2E_LAT]
        )

        all_metrics.append(request_metrics)
    completed_requests.extend(all_metrics)

    print(f"Results for token benchmark for {model} queried with the {llm_api} api.\n")
    ret = metrics_summary(completed_requests, start_time, end_time)

    metadata = {
        "model": model,
        "mean_input_tokens": mean_input_tokens,
        "stddev_input_tokens": stddev_input_tokens,
        "mean_output_tokens": mean_output_tokens,
        "stddev_output_tokens": stddev_output_tokens,
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

    overall_output_throughput = df_without_errored_req[
        common_metrics.NUM_OUTPUT_TOKENS
    ].sum() / (end_time - start_time)

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
    mean_input_tokens: int,
    stddev_input_tokens: int,
    mean_output_tokens: int,
    stddev_output_tokens: int,
    tokenizer: PreTrainedTokenizerFast,
    results_dir: str,
    user_metadata: Dict[str, Any],
    additional_sampling_params: Optional[str] = "",
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
    """
    if mean_input_tokens < 40:
        print(
            "the minimum number of input tokens that will be sent is 41"
            " because of the prompting logic right now"
        )

    summary, individual_responses = get_token_throughput_latencies(
        model=model,
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        max_num_completed_requests=max_num_completed_requests,
        mean_input_tokens=mean_input_tokens,
        stddev_input_tokens=stddev_input_tokens,
        mean_output_tokens=mean_output_tokens,
        stddev_output_tokens=stddev_output_tokens,
        num_concurrent_requests=num_concurrent_requests,
        additional_sampling_params={},
        tokenizer=tokenizer,
    )

    return summary, individual_responses
