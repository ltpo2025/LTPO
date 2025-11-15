import argparse
import numpy as np
import os
import random
import torch

from data import get_dataset
from extract_judge_answer import extract_answer, extract_true_answer, judge_answer
from transformers import AutoModelForCausalLM, AutoTokenizer
from ltpo import generate
from reward import RewardModel
from tqdm import tqdm


huggingface_token = os.environ['HUGGING_FACE_TOKEN']


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the model")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k", help="Dataset to evaluate")
    parser.add_argument("--model_name_or_path", type=str, help="Path to the model")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--start_data_idx", type=int, default=0, help="Start index of the data to evaluate")
    parser.add_argument("--end_data_idx", type=int, default=1319, help="End index of the data to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Number of generated tokens")
    parser.add_argument("--device", type=str, default="cuda")

    # prompt
    parser.add_argument("--solver_prompt_idx", type=int, default=0, help="Index of the solver prompt")

    # seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")

    # optimization args
    parser.add_argument('--num_thought_tokens', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--sigma_decay', type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=0.03, help="Learning rate")
    parser.add_argument("--max_num_steps", type=int, default=10, help="Number of optimization iterations")

    # reward model
    parser.add_argument("--reward_threshold", type=float, default=-1, help="Threshold for reward to stop optimization")
    parser.add_argument("--top_k", type=int, default=10, help="Use top-k most probable tokens to calculate token-level confidence")
    parser.add_argument("--disable_conf_reward", action="store_true", help="If set, disable using confidence reward")
    parser.add_argument("--disable_best_reward", action="store_true", help="If set, disable using best reward step as output")

    # misc
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--ckpt_suffix", type=str, default="")
    parser.add_argument("--use_auto_grad", action="store_true", help="Use PyTorch's auto-grad")
    parser.add_argument("--eval_baseline", action="store_true", help="Evaluate baseline")
    parser.add_argument("--verbose", type=int, default=1, help="Print detailed information")
    parser.add_argument("--disable_save_logistics", action="store_true", help="Disable saving the logistics.pt")
    return parser.parse_args()


def set_seed(seed):
    '''
    Set random seed for reproducibility

    Args:
        seed: random seed
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


# evaluate function 
def main(args):
    '''
    Evaluate model

    Args:
        dataset: dataset to evaluate
        sample_num: number of samples to evaluate

    Returns:
        original_accuracy: original generation accuracy
        optimized_accuracy: optimized generation accuracy
    '''

    if args.seed:
        set_seed(args.seed)

    # set device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        token=huggingface_token,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token=huggingface_token,
    )

    # load reward model
    reward_model = RewardModel(
        model=model,
        tokenizer=tokenizer,
        num_thought_tokens=args.num_thought_tokens,
    )

    # load dataset
    dataset = get_dataset(
        args.dataset, 
        tokenizer=tokenizer,
        prompt_idx=args.solver_prompt_idx,
    )
    if args.verbose:
        print(f"Example: {dataset[0]}")

    total = 0
    correct = 0
    entries = []
    model_name = args.model_name_or_path.split("/")[-1]
    data_name = args.dataset.split("/")[-1]
    conf_suffix = "" if args.disable_conf_reward else "-conf"

    if args.eval_baseline:
        model.eval()
        output_suffix = "-" + args.ckpt_suffix if args.ckpt_suffix else ""
        output_dir = f"{args.output_dir}/{model_name}-{data_name}-max_tokens{args.max_new_tokens}-prompt{args.solver_prompt_idx}" + output_suffix
    else:
        output_dir = f"{args.output_dir}/{model_name}-{data_name}-tokens{args.num_thought_tokens}-lr{args.lr}-sigma{args.sigma}-sigdecay{args.sigma_decay}" + conf_suffix

    start_data_idx = max(0, args.start_data_idx)
    end_data_idx = min(args.end_data_idx, len(dataset))

    if args.resume and not args.disable_save_logistics:
        print(f"Resume from {output_dir}")
        # load logistics
        logistics = torch.load(f"{output_dir}/logistics.pt")
        start_data_idx = logistics["start_idx"]
        correct = logistics["correct"]
        total = logistics["total"]
        entries = logistics["entries"]


    print(f"Start to evaluate {args.dataset} from {start_data_idx} to {end_data_idx}...")

    data_idx_list = range(start_data_idx, end_data_idx)
    for i in tqdm(data_idx_list):
        example = dataset[i]
        question = example['question']
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        true_answer = extract_true_answer(example["answer"], name=args.dataset)

        if args.verbose:
            print(f"Index {i}, Question: {question}")
            print(f"Index {i}, True answer: {true_answer}")
        if true_answer is None:
            continue

        if args.eval_baseline:
            best_reward, best_reward_step = None, None
            inputs = tokenizer.apply_chat_template(
                example["prompt"],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            outputs = model.generate(**inputs, **dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=None,
                num_beams=1,
            ))
            output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            output, best_reward, best_reward_step = generate(
                tokenizer=tokenizer,
                model=model,
                reward_model=reward_model,
                question=question,
                num_thought_tokens=args.num_thought_tokens,
                max_rl_steps=args.max_num_steps,
                max_new_tokens=args.max_new_tokens,
                reward_threshold=args.reward_threshold,
                lr=args.lr,
                sigma=args.sigma,
                sigma_decay=args.sigma_decay,
                use_auto_grad=args.use_auto_grad,
                disable_conf_reward=args.disable_conf_reward,
                disable_best_reward=args.disable_best_reward,
                data_name=args.dataset,
                model_name=args.model_name_or_path,
                verbose=args.verbose,
                top_k=args.top_k,
            )

        # extract answer from model response
        answer = extract_answer(
            output, 
            data_name=args.dataset, 
            prompt_idx=args.solver_prompt_idx, 
            model_name=args.model_name_or_path,
        )
        if args.verbose:
            if args.verbose > 1:
                print(f"Index {i}, LLM response:\n{output}")
            print(f"Index {i}, LLM answer: {answer}")
            print(f"Index {i}, True answer: {true_answer}")
            print(f"Index {i}, Best reward: {best_reward}, Best reward step: {best_reward_step}")

        # judge answer
        is_correct = False
        if answer is not None:
            is_correct = judge_answer(output, true_answer, data_name=args.dataset, prompt_idx=args.solver_prompt_idx)
            correct += is_correct

        if not args.disable_save_logistics:
            entries.append(dict(
                data_idx=i,
                question=question,
                response=output,
                answer=answer,
                is_correct=is_correct,
                best_reward=best_reward,
                best_reward_step=best_reward_step,
            ))

        total += 1
        
        # save logistics
        # save original correct, optimized correct, total, update count
        if not args.disable_save_logistics:
            torch.save({
                "start_idx": i+1,
                "total": total,
                "correct": correct,
                "entries": entries,
            }, f"{output_dir}/logistics.pt")
        print(f"Current state: correct={correct}, total={total}, accuracy={correct / total:.4f}")

    if args.verbose:
        for i, entry in enumerate(entries):
            if not entry['is_correct']:
                continue
            print(f"====================== Entry {i} ======================")
            print(f">>> Question:\n{entry['question']}")
            print(f">>> Response:\n{entry['response']}")
            print(f">>> Answer:\n{entry['answer']}")
            print(f">>> Data Idx: {entry['data_idx']}")
            print(f">>> Best Reward: {entry['best_reward']}, Best Reward Step: {entry['best_reward_step']}")

    print(f">>> Final State: correct={correct}, total={total}, accuracy={correct / total:.4f}")
    print(f">>> Data Idx with Correct Answer: {[entry['data_idx'] for entry in entries if entry['is_correct']]}")


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        print(f"-- {arg}: {getattr(args, arg)}")
    main(args)

