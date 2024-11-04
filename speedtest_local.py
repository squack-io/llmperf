import os
import ray
import pandas as pd
from transformers import AutoTokenizer
from datetime import datetime
import numpy as np
from src.llmperf.token_benchmark_ray import run_token_benchmark

# Create volumes
MODELS_DIR = "./llamas"
MODEL_NAME = "Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
CHOSEN_MODEL_DIR = "neuralmagic/" + MODEL_NAME


def main():
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

    # Set up Ray with environment variables
    env_vars = os.environ.copy()
    env_vars["OPENAI_API_KEY"] = (
        "f460c75b0ce07b8c2769b0219737a7ff78c163a12560ba3d3099184d403008b1"
    )
    env_vars["OPENAI_API_BASE"] = (
        "https://relace--quantization-test-serve.modal.run/v1/"
    )

    model_name = os.path.join(MODELS_DIR, CHOSEN_MODEL_DIR)

    # Define the tokenizer and set the padding parameters
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test parameters
    input_token_values = [1000, 2000, 4000, 8000, 16000, 32000]
    concurrent_request_values = [1, 2, 4, 8, 16, 32]
    results_list = []

    # Create timestamp for unique run identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "./benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    output_base_path = os.path.join(
        output_dir, f"benchmark_results_{MODEL_NAME}_{timestamp}"
    )

    # Run benchmarks for each combination
    for i, mean_input_tokens in enumerate(input_token_values):
        for j, num_concurrent_requests in enumerate(concurrent_request_values):
            print(f"\nCurrently running benchmark with:")
            print(f"Mean input tokens: {mean_input_tokens}")
            print(f"Number of concurrent requests: {num_concurrent_requests}")
            ray.init(runtime_env={"env_vars": env_vars})
            try:
                summary, _ = run_token_benchmark(
                    llm_api="openai",
                    model=MODEL_NAME,
                    test_timeout_s=2000,
                    max_num_completed_requests=32,
                    mean_input_tokens=mean_input_tokens,
                    stddev_input_tokens=5,
                    mean_output_tokens=100,
                    stddev_output_tokens=5,
                    num_concurrent_requests=num_concurrent_requests,
                    tokenizer=tokenizer,
                    text_input=(mean_input_tokens // 1000) * thousand_tokens,
                )
                results_list.append(summary)

                # Save intermediate results after each benchmark
                intermediate_df = pd.DataFrame(results_list)
                intermediate_path = f"{output_base_path}_intermediate.csv"
                intermediate_df.to_csv(intermediate_path, index=False)
            except Exception as e:
                print(f"Error during benchmark: {e}")
                continue
            ray.shutdown()

            print(
                f"Completed benchmark {i*len(concurrent_request_values) + j + 1} of {len(input_token_values)*len(concurrent_request_values)}"
            )
            print(f"Intermediate results saved to: {intermediate_path}")

    # Save final results
    final_df = pd.DataFrame(results_list)
    final_df = final_df.astype(float)
    final_path = f"{output_base_path}_complete.csv"
    final_df.to_csv(final_path, index=False)
    print(f"Final results saved to: {final_path}")


if __name__ == "__main__":
    main()
