import json
import utils
import os
import argparse
from tqdm import tqdm

def filter_file(file_name):
    print("Start filtering for file: " + file_name)
    
    data_file = f'./results/filtered/{file_name}'
    # read each line and get the question and options
    with open(data_file, 'r') as f:
        lines = f.readlines()
    dataset_name = file_name.split('.')[0].split('_')[0]
    
    target_file_name = f'./results/quality_sampling/{file_name}'
    if not os.path.exists('./results/quality_sampling'):
        os.makedirs('./results/quality_sampling')
    
    logger = utils.init_logger(name=dataset_name)
    
    with open(target_file_name, 'w') as f:
        for i in tqdm(range(len(lines))):
            sample = json.loads(lines[i])
            question = sample['question']
            options = sample['options'] if 'options' in sample.keys() else ''
            answer = sample['answer']
            reasoning = sample['reasoning']
            # remove the conclusion part in the reasoning, after 'Conclusion:'
            reasoning = reasoning.split('Conclusion:')[0]
            
            logger.info("Generating answer...")
            llm_answer = utils.llm_generate_answer_with_reasoning(question, options, reasoning, engine='gpt-4')
            logger.info(f'Q: {question}')
            logger.info(f'A: {answer}')
            logger.info(f'LLM-A: {llm_answer}')
            logger.info("Extracting answer...")
        
            logger.info("Judging answer...")
            judged_result = utils.llm_judge_answer(llm_answer, answer, engine='gpt-4')
            logger.info(f'Judged: {judged_result}')
            
            if "true" in judged_result.lower():
                # save the line
                f.write(lines[i])
                logger.info(f'True sample {i} saved.')
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file", type=str, required=True)
    args = parser.parse_args()
    source_file = args.source_file
    filter_file(source_file)