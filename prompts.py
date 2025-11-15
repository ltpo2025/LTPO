"""
Solver prompts for different math datasets.
"""

def gsm8k_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
        prompt_idx (int): The index of the prompt to be used.

    """     
    prompt = [
        # idx 0: boxed
        [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": q},
        ],
        # idx 1: json
        [
            {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
            {"role": "user", "content": 
                f"QUESTION: {q} \n"
                "Let's think step by step. "
                "Please provide your thought process and your final answer separately and response in json format "
                "containing the keys \"thought process\" and \"final answer\". "
                "For example your response should be "
                "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                "Note that the final answer should be pure numbers, not the calculation formulas, and without any units or explanation!!! "
            },
        ],
        # idx 2: no cot
        [
            {"role": "system", "content": "Please put your final answer within \\boxed{}."},
            {"role": "user", "content": q},
        ],
    ]
    return prompt[prompt_idx]


def asdiv_aug_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
        prompt_idx (int): The index of the prompt to be used.

    """     
    prompt = [
        # idx 0: boxed
        [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": q},
        ],
        # idx 1: json
        [
            {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
            {"role": "user", "content": 
                f"QUESTION: {q} \n"
                "Let's think step by step. "
                "Please provide your thought process and your final answer separately and response in json format "
                "containing the keys \"thought process\" and \"final answer\". "
                "For example your response should be "
                "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                "Note that the final answer should be pure numbers, not the calculation formulas, and without any units or explanation!!! "
            },
        ],
        # idx 2: no cot
        [
            {"role": "system", "content": "Please put your final answer within \\boxed{}."},
            {"role": "user", "content": q},
        ],
    ]
    return prompt[prompt_idx]


def math_500_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
    """
    prompt = [
        # idx 0: boxed
        [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": q},
        ],
        # idx 1: json
        [
            {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
            {"role": "user", "content": 
                f"QUESTION: {q} \n"
                "Let's think step by step. "
                "Please provide your thought process and your final answer separately and response in json format "
                "containing the keys \"thought process\" and \"final answer\". "
                "For example your response should be "
                "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
            },
        ],
        # idx 2: no cot
        [
            {"role": "system", "content": "Please put your final answer within \\boxed{}."},
            {"role": "user", "content": q},
        ],
    ]
    return prompt[prompt_idx]


def aime_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
        prompt_idx (int): The index of the prompt to be used.

    """     
    prompt = [
        # idx 0: boxed
        [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": q},
        ],
        # idx 1: json 
        [
            {"role": "system", "content": "You are a precise math question solver. Solve this math problem. "},
            {"role": "user", "content": 
                f"QUESTION: {q} \n"
                "Let's think step by step. "
                "Please provide your thought process and your final answer separately and response in json format "
                "containing the keys \"thought process\" and \"final answer\". "
                "For example your response should be "
                "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                "Note that the final answer should be pure numbers, not the calculation formulas, and without any units or explanation!!! "
            },
        ],
        # idx 2: no cot
        [
            {"role": "system", "content": "Please put your final answer within \\boxed{}."},
            {"role": "user", "content": q},
        ],
    ]
    return prompt[prompt_idx]

def strategy_qa_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
        prompt_idx (int): The index of the prompt to be used.

    """
    prompt = [
        # idx 0: boxed
        [
            {"role": "system", "content": "Please reason step by step, and answer the following question with `Yes` or `No`."},
            {"role": "user", "content": q},
        ],
        # idx 1: json
        [
            {"role": "system", "content": "You are required to answer the following question with `Yes` or `No`."},
            {"role": "user", "content":
                f"QUESTION: {q} \n"
                "Let's think step by step. "
                "Please provide your thought process and your final answer separately and response in json format "
                "containing the keys \"thought process\" and \"final answer\". "
                "For example your response should be "
                "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                "Note that the final answer should be pure `Yes` or `No`, without any details or explanation!!! "
            },
        ],
        # idx 2: no cot
        [
            {"role": "system", "content": "You are required to answer the following question with `Yes` or `No`."},
            {"role": "user", "content": q},
        ],
    ]
    return prompt[prompt_idx] 

def du_prompt(q, prompt_idx=0):
    """
    Args:
        q (str): The question to be solved.
        prompt_idx (int): The index of the prompt to be used.

    """
    prompt = [
        # idx 0: boxed
        [
            {"role": "system", "content":
                "Please reason step by step, and put your final answer within \\boxed{}. "
                "In the multiple choices problem, there are five options: A, B, C, D, E, and F, respectively. "
                "The correct answer that solves the problem is one of these options. "
                "Only one letter from A to F is accepted in the answer span."
            },
            {"role": "user", "content": q},
        ],
        # idx 1: json
        [
            {"role": "system", "content":
                "You are a precise math question solver. Solve this multiple choices problem. "
                "In the multiple choices problem, there are five options: A, B, C, D, E, and F, respectively. "
                "The correct answer that solves the problem is one of these options. "
                "Only one letter from A to F is accepted in the answer span."
            },
            {"role": "user", "content":
                f"QUESTION: {q} \n"
                "Let's think step by step. "
                "Please provide your thought process and your final answer separately and response in json format "
                "containing the keys \"thought process\" and \"final answer\". "
                "For example your response should be "
                "{\"thought process\": \"your thought process\", \"final answer\": \"your final answer\"}. "
                "Note that the final answer should be pure option letter, not the calculation formulas,  without any details or explanation!!! "
            },
        ],
        # idx 2: no cot
        [
            {"role": "system", "content":
                "Please put your final answer within \\boxed{}. "
                "In the multiple choices problem, there are five options: A, B, C, D, E, and F, respectively. "
                "The correct answer that solves the problem is one of these options. "
                "Only one letter from A to F is accepted in the answer span."
            },
            {"role": "user", "content": q},
        ],
    ]
    return prompt[prompt_idx]
