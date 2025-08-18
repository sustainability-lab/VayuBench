from code_eval_utils import evaluate_code_set
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import argparse
import logging
import json
import os 
import time

# ! Accepting Model name from terminal
parser = argparse.ArgumentParser(description="Running this model result generation for eval questions.")
parser.add_argument("--model", type=str, required=True, help="The name of Model on Huggingface")
parser.add_argument("--starts", type=int, required=True, help="The number of response for starting result creation")
parser.add_argument("--ends", type=int, required=True, help="The number of response for ending result creation")
args = parser.parse_args()

model_name = args.model

# ! Assinging Constants 
SAMPLE_SIZE = 5
base_filepath = os.getcwd()

# ! Configuring Logs 
log_filename = f"{base_filepath}/chunk_result_logs.txt"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_message(message):
    print(message, flush=True)
    logging.info(message)

log_message("Pipeline started.")

df = pd.read_json(f'/home/ubuntu/eval/{model_name}/response.json')
starts = args.starts
ends = args.ends
df = df.loc[starts - 1 : ends - 1]
categories = df["category"].unique().tolist()
model_responses = []

for i in range(starts, ends+1):
    for _, file in enumerate(glob(f'{base_filepath}/responses/{model_name}/*/{i}/*')):
        with open(file, 'r') as f:
            data = json.load(f)
            model_responses.append(data['generated_samples'].copy())
df['generated_samples'] = model_responses

def pass_at_K_on_df(n, df):
    results = []
    for _, row in df.iterrows():
        id = row["id"]
        answer = row["answer"]
        question = row["question"]
        category = row["category"]
        code = row["code"]
        sample = row["generated_samples"][:SAMPLE_SIZE]
        refer = textwrap.dedent(f"""
            import pandas as pd
            import numpy as np
            df = pd.read_pickle("{base_filepath}/preprocessed/main_data.pkl")
            ncap_funding_df = pd.read_pickle("{base_filepath}/preprocessed/ncap_funding_data.pkl")
            states_df = pd.read_pickle("{base_filepath}/preprocessed/states_data.pkl")
            output = get_response(df, states_df, ncap_funding_df)
            print('EXEC_OK:', output, flush=True)
            try:
                assert np.isclose(float(output),float({repr(answer)}))
            except (ValueError, TypeError):
                assert output == {repr(answer)}
        """)
        
        question_folder = f"{base_filepath}/results_chunk/{model_name}/{starts}_{ends}/{id}"
        question_file_path = f"{question_folder}/result.json"
        os.makedirs(question_folder, exist_ok=True)
        
        if os.path.isfile(question_file_path):
            log_message(f"Skipping question {id} as result already exists.")
            existing_results = pd.read_json(question_file_path).to_dict(orient="records")
            results.extend(existing_results)
            continue
        start_time = time.time()
        pass_at_k, result = evaluate_code_set(
            test_case=refer,
            candidates=sample,
            num_workers = 12,
            k_values=n,
            timeout=15)
        end_time = time.time()
        question_results = []
        for i in range(SAMPLE_SIZE):
            res = {
                'id': id,
                'question': question,
                'answer': answer,
                'category': category,
                'model': model_name,
                'true_code': code,
                'pass@1': pass_at_k['pass@1'],
                'pass@2': pass_at_k['pass@2'],
                'exec@1': pass_at_k['exec@1'],
                'executed': result[i]['executed'],
                'passed': result[i]['passed'],
                'code_out': result[i]['code_out'],
                'error_type': result[i]['error_type'],
                'traceback': result[i]['traceback'],
                'sample': sample[i],
            }
            question_results.append(res)
            results.append(res.copy())
            log_message(res)
        log_message(f"Time taken for {SAMPLE_SIZE} responses:{end_time-start_time} seconds.")

        pd.DataFrame(question_results).to_json(question_file_path, orient="records", indent=4)

    return results
# !
def save_results(df, file_path):
    result_data = pass_at_K_on_df([1, 2], df)
    result_df = pd.DataFrame(result_data)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    result_df.to_json(file_path, orient="records", indent=4)
    return result_df

# ! 
aggregated_results_df = save_results(df,f"{base_filepath}/results_chunk/{model_name}/{starts}_{ends}/result.json")

os.makedirs(os.path.dirname(f"{base_filepath}/results_chunk/{model_name}/{starts}_{ends}/result.json"), exist_ok=True)
aggregated_results_df.to_json(f"{base_filepath}/results_chunk/{model_name}/{starts}_{ends}/result.json", orient="records", indent=4)

# ! PLOT
# Compute mean and SEM
metrics = aggregated_results_df.drop(columns=['id', 'question', 'answer', 'model','true_code', 'executed', 'passed', 'code_out', 'error_type', 'traceback', 'sample'])
mean_result = metrics.groupby("category").mean().reset_index()
sem_result = metrics.groupby("category").sem().reset_index()

# Sort both by mean pass@1
mean_result = mean_result.sort_values(by="pass@1", ascending=False)
sem_result = sem_result.loc[mean_result.index]  # ensure same order

pass_k_s = ["pass@1", "pass@2", "exec@1"]
categories = mean_result["category"].tolist()
num_categories = len(categories)

x = np.arange(num_categories)
bar_width = 0.25
colors = plt.get_cmap('Set2').colors

fig, ax = plt.subplots(figsize=(14, 7))

for i, pass_k in enumerate(pass_k_s):
    means = mean_result[pass_k]
    sems = sem_result[pass_k]
    ax.bar(x + i * bar_width, means, yerr=sems, width=bar_width,
           label=pass_k, color=colors[i % len(colors)], capsize=4, edgecolor='black')

ax.set_xticks(x + bar_width)
ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=10)

ax.set_xlabel("Category", fontsize=12)
ax.set_ylabel("Metric Value", fontsize=12)
ax.set_title(f"Evaluation Metrics with SEM for Each Category - {model_name}", fontsize=14)

ax.legend(title="Metrics")
ax.grid(axis='y', linestyle='--', alpha=0.7)

save_path = f"{base_filepath}/chunk_charts/{model_name}/{starts}_{ends}_pass_k_results.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
plt.show()