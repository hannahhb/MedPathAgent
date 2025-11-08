import json
import os
from tqdm import tqdm
import yaml
from dataset import QADataset

# recover the data from the log file
def recover_data_from_log(log_path, target_file):
    with open(log_path, 'r') as f:
        lines = f.readlines()
        # find the beginning of each data, containing the INFO - Processing nth sample, where n is a number
        data_start_lines = [i for i, line in enumerate(lines) if "INFO - Processing" in line]
        # data end lines is the next data start line, for last data, it is the end of the file
        data_end_lines = data_start_lines[1:] + [len(lines)]
        data = []
        # get the data between each start and end line
        for start, end in zip(data_start_lines, data_end_lines):
            current_data = lines[start:end]
            # get the sample id from the start line, for example, "2025-02-17 14:08:10,249 - INFO - Processing 0th sample.", then sample id is 1
            sample_id = int(current_data[0].split(" ")[-2][:-2])
            print(f"Processing {sample_id}th sample")
            # find the question and answer lines, containing "INFO - Answer:" and "INFO - Question:" respectively
            question_line = [line for line in current_data if "INFO - Question:" in line][0]
            answer_line = [line for line in current_data if "INFO - Answer:" in line][0]
            reasoning_line = [line for line in current_data if "INFO - Reasoning:" in line][0]
            # question content begins from the question line and ends at the answer line, concat all these lines into the question content
            question_content = ''.join(current_data[current_data.index(question_line):current_data.index(answer_line)])
            
            answer_line_ends = [line for line in current_data if "INFO - Total API cost so far (accumulated in run_llm calls):" in line][0]
            answer_content = ''.join(current_data[current_data.index(answer_line):current_data.index(answer_line_ends)])
            reasoning_content = ''.join(current_data[current_data.index(reasoning_line):])
            
            
            # get the question and answer from the line, which is the text after "INFO - Question:" and "INFO - Answer:"
            
            question = question_content.split("INFO - Question:")[-1].strip()
            answer = answer_content.split("INFO - Answer:")[-1].strip()
            reasoning = reasoning_content.split("INFO - Reasoning:")[-1].strip()
            
            data.append({
                "id": sample_id,
                "question": question,
                "reasoning": reasoning,
                "huatuo": "",
                "answer": answer,
            })
        # write the data to the target jsonl file
        with open(target_file, 'w') as f:
            for sample in data:
                f.write(json.dumps(sample) + '\n')
                
# get the content format to training Qwen
def process_cot_example_Qwen(
    example: dict,
    tokenizer,
):
    QUERY_TEMPLATE_NOANSWER = """{Question}""".strip()
    reasoning = example["reasoning"]
    question = example["question"]
    answer = example["answer"]
    prompt = QUERY_TEMPLATE_NOANSWER.format(Question=question)
    answer = "Answer: " + answer if "Answer:" not in answer else answer
    text = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
        {
            "role": "assistant", 
            "content": "<|im_start|>think\n" + reasoning.strip() + "\n<|im_start|>answer\n" + answer.strip()
        }
    ], tokenize=False)
    return dict(
        dataset_name=example["dataset_name"],
        id_in_dataset=example["id_in_dataset"],
        text=text
    )

            
def failed_data_filtering(
    source_file: str,
    target_file: str,
):
    print(f"Filtering the data for {source_file}...")
    # open the source jsonl file
    with open(source_file, 'r') as f:
        lines = f.readlines()
        # load the jsonl file
        data = [json.loads(line) for line in lines]
    # filter the data
    filtered_data = []
    for example in tqdm(data):
        if "No reasoning path" in example["reasoning"]:
            continue
        # rename the key "id" to "id_in_dataset"
        dataset_name = source_file.split("/")[-1].split(".")[0].split("_")[:-1]
        dataset_name = "_".join(dataset_name)
        filtered_data.append({
            "dataset_name": dataset_name,
            "id_in_dataset": example["id"],
            "question": example["question"],
            "answer": example["answer"],
            "reasoning": example["reasoning"],
            "options": example["options"] if "options" in example else ''
        })
    # write the filtered data to the target jsonl file
    # print the number of filtered data
    print(f"Filtered {len(data) - len(filtered_data)} samples from {source_file}.")
    with open(target_file, 'w') as f:
        for example in filtered_data:
            f.write(json.dumps(example) + '\n')
            
def merge_qwen_files(
    qwen_data_floder: str,
    target_file: str,
):
    # get all the files in the qwen_data_floder
    file_list = os.listdir(qwen_data_floder)
    # filter the files that are jsonl files
    file_list = [file for file in file_list if file.endswith(".jsonl")]
    # open the target jsonl file
    with open(target_file, 'w') as f:
        for file in file_list:
            # open the source jsonl file
            with open(qwen_data_floder + file, 'r') as f_source:
                lines = f_source.readlines()
                # load the jsonl file
                data = [json.loads(line) for line in lines]
                for example in data:
                    f.write(json.dumps(example) + '\n')
    print(f"Merged {len(file_list)} files into {target_file}.")
    
def get_options_for_data(
    data_files: str,
    target_file: str
):
    with open('./dataset_configs.yml', 'r') as f:
        dataset_configs = yaml.safe_load(f)
    # open the data jsonl file, read each lines
    with open(data_files, 'r') as f:
        lines = f.readlines()
        # load the jsonl file
        data_list = [json.loads(line) for line in lines]
        dataset_name = data_list[0]["dataset_name"]
    dataset = QADataset(**dataset_configs[dataset_name])
    # get the options for each data
    for data in tqdm(data_list):
        idx = data["id_in_dataset"]
        data_smaple = dataset[idx]
        data["options"] = data_smaple["options"]
    # write the data to the target jsonl file
    with open(target_file, 'w') as f:
        for example in data_list:
            f.write(json.dumps(example) + '\n')
    print(f"Saved the data with options to {target_file}.")
    return


def get_intersection_data(file_a, file_b, file_target):
    # read two jsonl files, only keep the data that are in both files
    with open(file_a, 'r') as f:
        lines_a = f.readlines()
        data_a = [json.loads(line) for line in lines_a]
    with open(file_b, 'r') as f:
        lines_b = f.readlines()
        data_b = [json.loads(line) for line in lines_b]
    # get the intersection data
    intersection_data = []
    for data in data_a:
        if data in data_b:
            intersection_data.append(data)
    # write the intersection data to the target jsonl file
    with open(file_target, 'w') as f:
        for example in intersection_data:
            f.write(json.dumps(example) + '\n')
    print(f"Saved the intersection data to {file_target}.")
    return