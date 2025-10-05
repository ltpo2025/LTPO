"""
Data API
"""
import json

from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
from prompts import gsm8k_prompt, asdiv_aug_prompt, math_500_prompt, aime_prompt, strategy_qa_prompt, du_prompt

def get_dataset(data_name_or_path, tokenizer, prompt_idx):
    """
    Args:
        data_name_or_path: dataset name or path
        tokenizer: tokenizer
        prompt_idx: which query prompt to use
    Returns:
        dataset: dataset
    """

    ### Load dataset ### 
    if "gsm8k" in data_name_or_path.lower():
        try:
            dataset = load_from_disk(data_name_or_path)['test']
        except:
            dataset = load_dataset("openai/gsm8k", "socratic")["test"]
        question_col = "question"
        answer_col = "answer"
    
    elif "asdiv-aug" in data_name_or_path.lower():
        try:
            dataset = load_from_disk(data_name_or_path)['test']
        except:
            dataset = load_dataset("xuyige/ASDiv-Aug")["test"]
        question_col = "question"
        answer_col = "answer"

    elif "math-500" in data_name_or_path.lower():
        try:
            dataset = load_from_disk(data_name_or_path)['test']
        except:
            dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        question_col = "problem"
        answer_col = "answer"

    elif "aime_2024" in data_name_or_path.lower():
        try:
            dataset = load_from_disk(data_name_or_path)
        except:
            dataset = load_dataset("Maxwell-Jia/AIME_2024")['train']
        question_col = "Problem"
        answer_col = "Answer"
    
    elif "aime2025" in data_name_or_path.lower():
        try:
            dataset = load_from_disk(data_name_or_path)
        except:
            dataset = concatenate_datasets([
                load_dataset("opencompass/AIME2025", "AIME2025-I")['test'],
                load_dataset("opencompass/AIME2025", "AIME2025-II")['test'],
            ])
        question_col = "question"
        answer_col = "answer"

    elif "strategyqa" in data_name_or_path.lower():
        return get_strategyqa(tokenizer, prompt_idx)

    elif "date_understanding" in data_name_or_path.lower():
        dataset = load_dataset("maveriq/bigbenchhard", "date_understanding")['train']
        question_col = "input"
        answer_col = "target"

    else:
        raise ValueError(f"Unsupported dataset: {data_name_or_path}")

    # preprocess dataset
    def preprocess_function(examples):
        '''
        Preprocess dataset

        Args:
            examples: dataset examples

        Returns:
            formatted: formatted dataset
        '''
        prompt = []
        formatted = []
        answers = examples[answer_col]
        questions = examples[question_col]
        for q in questions:
            if "gsm8k" in data_name_or_path.lower():
                messages = gsm8k_prompt(q, prompt_idx)
            elif "asdiv-aug" in data_name_or_path.lower():
                messages = asdiv_aug_prompt(q, prompt_idx)
            elif "math-500" in data_name_or_path.lower():
                messages = math_500_prompt(q, prompt_idx)
            elif "aime" in data_name_or_path.lower():
                messages = aime_prompt(q, prompt_idx)
            elif "date_understanding" in data_name_or_path.lower():
                messages = du_prompt(q, prompt_idx)
            else:
                raise ValueError(f"Unsupported dataset: {data_name_or_path}")

            prompt.append(messages)
            formatted.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        if "aime" in data_name_or_path.lower() and "2025" in data_name_or_path.lower():
            answers = [ans.replace('^\circ', '') for ans in answers]
        if "date_understanding" in data_name_or_path.lower():
            answers = [ans[1] for ans in answers]
        return {
            "prompt": prompt,
            "formatted": formatted,
            "question": questions,
            "answer": answers,
        }

    dataset = dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
    return dataset

def get_strategyqa(tokenizer, prompt_idx):
    prompt = []
    formatted = []
    answers = []
    questions = []
    with open('strategyqa_train.json', 'r') as f:
        data = json.load(f)
    for ins in data:
        q, a = ins['question'], ins['answer']
        msg = strategy_qa_prompt(q, prompt_idx)
        questions.append(q)
        answers.append(a)
        prompt.append(msg)
        formatted.append(tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        ))
    return Dataset.from_dict({
        "prompt": prompt,
        "formatted": formatted,
        "question": questions,
        "answer": answers,
    })
