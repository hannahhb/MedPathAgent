from datasets import load_dataset
from torch.utils.data import Dataset
import re

# now get a parser class for the dataset, unify the getting of data interface
class QADataset(Dataset):
    def __init__(self, file_type, path, parsers, split='train', **kwargs):
        self.parsers = parsers

        if file_type == 'huggingface':
            # HuggingFace hub datasets support split directly
            self.ds = load_dataset(path, split=split, **kwargs)
        else:
            # For local files (e.g., json/jsonl/csv), ensure the requested split exists.
            # If `path` is a string, map it to the requested split name (train/test/validation).
            if isinstance(path, dict):
                # User provided explicit mapping like {'test': '/path/to.jsonl'}
                self.ds = load_dataset(file_type, data_files=path, **kwargs)[split]
            else:
                # Single file path: create a split with the requested name
                self.ds = load_dataset(file_type, data_files={split: path}, **kwargs)[split]

    def __len__(self):
        return len(self.ds)

    def default_parser(self, keys: list[dict], row: dict):
        # If an empty list is passed (e.g., [] in YAML), return empty string
        if not keys:
            return ""
        parts = []
        for part in keys:
            val = row[part['key']]
            # ensure string
            parts.append(part['prefix'] + str(val) + part['suffix'])
        return ' '.join(parts)

    # "choices": ["six weeks post-fertilization.", "eight weeks post-fertilization.", ...]
    def mmlu_option_parser(self, row: dict):
        choices = row['choices']
        context = '\n'.join([f'{chr(ord("A") + i)}. {choice}' for i, choice in enumerate(choices)])
        return "Answer Choices:\n" + context

    # "options": {"A": "...", "B": "...", "C": "...", "D": "..."}
    def medqa_option_parser(self, row: dict):
        options = row['options']
        context = '\n'.join([f'{key}. {option}' for key, option in options.items()])
        return "Answer Choices:\n" + context

    # four columns: opa, opb, opc, opd
    def medbullets_op4_option_parser(self, row: dict):
        options = [row['op' + chr(ord('a') + i)] for i in range(4)]
        context = '\n'.join([f'{chr(ord("A") + i)}. {option}' for i, option in enumerate(options)])
        return "Answer Choices:\n" + context

    # five columns: opa..ope
    def medbullets_op5_option_parser(self, row: dict):
        options = [row['op' + chr(ord('a') + i)] for i in range(5)]
        context = '\n'.join([f'{chr(ord("A") + i)}. {option}' for i, option in enumerate(options)])
        return "Answer Choices:\n" + context

    # four columns: opa..opd
    def medmcqa_option_parser(self, row: dict):
        options = [row['op' + chr(ord('a') + i)] for i in range(4)]
        context = '\n'.join([f'{chr(ord("A") + i)}. {option}' for i, option in enumerate(options)])
        return "Answer Choices:\n" + context

    # dict-based
    def medxpertqa_option_parser(self, row: dict):
        options = row['options']
        context = '\n'.join([f'{key}. {option}' for key, option in options.items()])
        return "Answer Choices:\n" + context

    # two options yes/no
    def pubmedqa_option_parser(self, row: dict):
        return "Answer Choices:\nA. Yes\nB. No"

    def medxpertqa_answer_parser(self, row: dict):
        options = row['options']
        answer_label = row['label']
        return f'({answer_label}) ' + options[answer_label]

    def mmlu_answer_parser(self, row: dict):
        answer = row['answer']        # e.g., "a" or ["a","c"]
        choice = row['choices']       # list[str]
        assert len(choice) == 4
        if isinstance(answer, str):
            answers_id = [ord(answer.lower()) - ord('a')]
        else:
            answers_id = [ord(ans.lower()) - ord('a') for ans in answer]
        result = ' And '.join([choice[i] for i in answers_id])
        return result

    def medmcqa_answer_parser(self, row: dict):
        answer_id = row['cop']  # int 0..3
        result = row['op' + chr(ord('a') + answer_id)]
        exp = row.get('exp')
        if exp:
            result = result + '. Explanation: ' + exp
        return result

    # === NEW: CURE-Bench parser (dict-based options like {"A": "...", ...}) ===
    def curebench_option_parser(self, row: dict):
        options = row['options']
        context = '\n'.join([f'{key}. {option}' for key, option in options.items()])
        return "Answer Choices:\n" + context

    def __getitem__(self, idx):
        # read the raw row
        raw_data = {}
        for key in self.ds.column_names:
            raw_data[key] = self.ds[key][idx]

        result = {}

        for component in ['question', 'answer', 'comparison','options']:
            parser = self.parsers.get(component, [])
            # if parser is a list, use the default parser
            if isinstance(parser, list):
                result[component] = self.default_parser(parser, raw_data)
            # else the parser is a string, call the function with the string name, which is a function in the class
            elif isinstance(parser, str):
                func = getattr(self, parser)
                result[component] = func(raw_data)
            else:
                # if parser is empty / None -> empty string
                result[component] = ""

        # --- NEW: preserve original id if present in the source row, else use idx ---
        # Ensure the id is returned as a string to match original datasets
        if 'id' in raw_data and raw_data['id'] is not None:
            result['id'] = str(raw_data['id'])
        else:
            result['id'] = str(idx)

        return result
