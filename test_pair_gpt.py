from pair import pair_algorithm
import pandas as pd
import json

results = []
benchmark_df = pd.read_csv('behaviors_benchmark.csv')

# Record the results of running PAIR into a dictionary for each objective
for _, row in benchmark_df.iterrows():
    row_dict = {
        "Objective": row["Goal"],
        "Category": row["Category"],
        "Output": pair_algorithm(num_iterations=4, attack_objective=row["Goal"], start_string=row["Target"])
    }
    # Append the dictionary to the result list
    results.append(row_dict)

with open("results/gpt_results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to results/gpt_results.json")