import argparse
import logging
import json

# ! Accepting Model name from terminal

parser = argparse.ArgumentParser(description="Running this model for eval questions.")
parser.add_argument("--model", type=str, required=True, help="The name of Model on Huggingface")
args = parser.parse_args()

model_name = args.model
print(f"RESPONSE GENERATION STARTED FOR MODEL : {model_name}")

# ! Loading model to GPU

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

device = torch.device('cuda')
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ! Assinging Constants 

SAMPLE_SIZE = 10
BATCH_SIZE = 10

import os 
base_filepath = os.getcwd()

# ! Read system prompt from the default file

SYSTEM_PROMPT_FILE = "system_prompt.txt"

if not os.path.exists(SYSTEM_PROMPT_FILE):
    raise FileNotFoundError(f"System prompt file '{SYSTEM_PROMPT_FILE}' not found.")

with open(SYSTEM_PROMPT_FILE, "r") as prompt_file:
    system = prompt_file.read().strip()

# ! Configuring Logs 

log_filename = f"{base_filepath}/pipeline_logs.txt"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log_message(message):
    print(message, flush=True)
    logging.info(message)
log_message("Pipeline started.")

# !

import pandas as pd
df = pd.read_csv("questions.csv")

categories = df["category"].unique().tolist()

# !

def post_process(code):
    ans = []
    for ins in code :
        if '<think>' in ins:
            parts = ins.split('</think>', 1)
            ins = parts[1] if len(parts) > 1 and parts[1].strip() else ""
        ins = ins.split('</code>')[0]
        ins = ins.replace('```python', '')
        ins = ins.split('```')[0]
        ins = ins.replace('<code>', '')
        ans.append(ins)
    return ans

# !

def sample_responses(batch_questions, batch_ids, category, sample=SAMPLE_SIZE):
    all_inputs = []
    input_lengths = []

    for question in batch_questions:
        chat = [
            {"role": "system", "content": system},
            {"role": "user", "content": question}
        ]
        formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        tokenized = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)
        input_len = tokenized["input_ids"].shape[1]
        all_inputs.extend([formatted] * sample)
        input_lengths.extend([input_len] * sample)
    
    batch = tokenizer(
        all_inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=False
    ).to(device)

    outputs = model.generate(
        **batch,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        temperature=0.7
    )

    decoded = [
        tokenizer.decode(output[input_len:], skip_special_tokens=True)
        for output, input_len in zip(outputs, input_lengths)
    ]

    responses = []

    for j, batch_id in enumerate(batch_ids):

        single_que_responses = decoded[j * SAMPLE_SIZE:(j + 1) * SAMPLE_SIZE]

        data = {
            "id": batch_id,
            "generated_samples": post_process(single_que_responses)
        }

        responses.append(data.copy())
        
        save_path = f"{base_filepath}/responses/{model_name}/{category}/{batch_id}/response.json"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(data, f, indent=4)

        log_message(f"Resopnse Generation : \nC ategory : \"{category}\" completed for Question : \"{batch_id}\"")
        
    return responses

# !

import os

for category in categories:

    file_path = f"{base_filepath}/responses/{model_name}/{category}/response.json"

    # Skip the category if already processed
    if os.path.isfile(file_path):
        log_message(f"Response Generation : \nCategory : \"{category}\" completed for Model : \"{model_name}\"")
        continue

    temp_df = df[df["category"] == category].reset_index(drop=True)
    model_responses = []

    # Collect already processed responses
    for idx, row in temp_df.iterrows():
        id = row['id']
        que_file_path = f"{base_filepath}/responses/{model_name}/{category}/{id}/response.json"
        
        if os.path.isfile(que_file_path):
            with open(que_file_path, "r") as f:
                data = json.load(f)
                model_responses.append(data.copy())
            log_message(f"Response Generation : \nCategory : \"{category}\" completed for Question : \"{id}\"")
        else:
            # Start batch processing from the first unprocessed question
            work_df = temp_df[idx:].reset_index(drop=True)
            break
    else:
        # If all questions in the category are processed, skip batch processing
        work_df = pd.DataFrame()

    # Process remaining unprocessed questions
    if not work_df.empty:
        for i in range(0, len(work_df), BATCH_SIZE):
            batch_df = work_df.iloc[i:i + BATCH_SIZE]
            batch_questions = batch_df['question'].tolist()
            batch_ids = batch_df['id'].tolist()
            batch_data = sample_responses(batch_questions, batch_ids, category, sample=SAMPLE_SIZE)
            model_responses.extend(batch_data)

    # Save responses for the entire category
    response_df = pd.DataFrame(model_responses)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    response_df.to_json(file_path, orient="records", indent=4)

    log_message(f"Response Generation : \nCategory : \"{category}\" saved to '{file_path}'")