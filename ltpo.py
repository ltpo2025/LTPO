import torch

from fastNLP import logger
from reward import RewardModel


def build_inputs(
    tokenizer,
    num_thought_tokens: int,
    prompt: str,
    device: str = 'cuda',
    data_name: str = '',
    model_name: str = '',
):
    if num_thought_tokens <= 0:
        raise ValueError('You must specify a positive int for num_thought_tokens')

    if 'llama' in model_name.lower():
        latent_thought_tokens = ''.join(f'<|reserved_special_token_{i}|>' for i in range(num_thought_tokens))
        gen_prompt = '<|start_header_id|>assistant<|end_header_id|>\n\n'
    elif 'qwen' in model_name.lower():
        latent_thought_tokens = '<|endoftext|>' * num_thought_tokens
        gen_prompt = '<|im_start|>assistant\n'
    elif 'mistral' in model_name.lower():
        latent_thought_tokens = '<unk>' * num_thought_tokens
        gen_prompt = ''
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    problem_type = 'code reasoning' if 'cruxeval' in data_name else 'math'

    if 'strategyqa' in data_name.lower():
        input_content = (
            f'Solve the following reasoning problem step by step.\n'
            f'You are required to answer the question with `Yes` or `No`.\n'
            f'PROBLEM: {prompt}\n\n'
            f'There are {num_thought_tokens} special tokens that contain compressed latent reasoning information '
            f'that might be useful for your reasoning.\n'
            f'If these tokens are useful for your case, you can use them as reference. If these tokens are not useful '
            f'for your case, you can ignore them and focus back to solving the problem.\n\n'
            f'Here are the {num_thought_tokens} special tokens: {latent_thought_tokens}'
        )
    elif 'date_understanding' in data_name.lower():
        input_content = (
            f'Solve the following multiple choices problem step by step.\n'
            f'In the multiple choices problem, there are five options: A, B, C, D, E, and F, respectively.\n'
            f'The correct answer that solves the problem is one of these options.\n'
            f'Your job is to solve the problem and find the correct option.\n'
            f'IMPORTANT: you MUST always put your final answer within \\boxed{{answer}}, '
            f'where [answer] is the option from A, B, C, D, E, and F. '
            f'Only one letter from A to F is accepted in the answer span.\n\n'
            f'PROBLEM: {prompt}\n\n'
            f'There are {num_thought_tokens} special tokens that contain compressed latent reasoning information '
            f'that might be useful for your reasoning.\n'
            f'If these tokens are useful for your case, you can use them as reference. If these tokens are not useful '
            f'for your case, you can ignore them and focus back to solving the problem.\n\n'
            f'Here are the {num_thought_tokens} special tokens: {latent_thought_tokens}'
        )
    else:
        input_content = (
            f'Solve the following {problem_type} problem efficiently and clearly:\n'
            f'- For simple problems (2 steps or fewer):\n'
            f'Provide a concise solution with minimal description.\n'
            f'- For complex problems (3 steps or more):\n'
            f'Use this step-by-step format:\n\n'
            f'## Step 1: [Brief calculations]\n'
            f'## Step 2: [Brief calculations]\n'
            f'...\n'
            f'IMPORTANT: Regardless of the approach, you MUST always put your final answer within \\boxed{{}}.\n\n'
            f'PROBLEM: {prompt}\n\n'
            f'There are {num_thought_tokens} special tokens that contain compressed latent reasoning information '
            f'that might be useful for your reasoning.\n'
            f'If these tokens are useful for your case, you can use them as reference. If these tokens are not useful '
            f'for your case, you can ignore them and focus back to solving the problem.\n\n'
            f'Here are the {num_thought_tokens} special tokens: {latent_thought_tokens}'
        )

    input_messages = [{'role': 'user', 'content': input_content}]
    inputs = tokenizer.apply_chat_template(
        input_messages, add_generation_prompt=False, return_dict=True, return_tensors="pt",
    ).to(device)

    # calculate the start index and the end index of the latent thought tokens.
    input_ids = inputs['input_ids'][0]
    pure_input_length = len(input_ids)
    input_thought_start_idx = pure_input_length - 1 - num_thought_tokens
    if 'qwen' in model_name.lower():
        input_thought_start_idx -= 1
    input_thought_end_idx = input_thought_start_idx + num_thought_tokens
    thought_idx = [input_thought_start_idx, input_thought_end_idx]

    if 'mistral' not in model_name.lower():
        gen_prompt_ids = torch.tensor(
            tokenizer.encode(gen_prompt, add_special_tokens=False),
            dtype=torch.long,
        ).unsqueeze(0).to(device)
        # Append the generation_prompt tokens to inputs['input_ids']
        inputs['input_ids'] = torch.cat([inputs['input_ids'], gen_prompt_ids], dim=1)
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'], device=device)

    return inputs, thought_idx

def get_confidence(
    model,
    inputs,
    thought_idx,
    thought_hidden_states,
    k=10,
):
    inputs['inputs_embeds'][0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
    logits = model(**inputs, return_dict=True)['logits'][0]
    probs = torch.softmax(logits, dim=-1)
    confidence = 0.0
    for idx in range(thought_idx[0], thought_idx[1] + 1):
        topk = torch.topk(probs[idx], k=k, largest=True)[0]
        confidence -= torch.sum(torch.log(topk + 1e-10)) / k
    num_tokens = thought_idx[1] - thought_idx[0] + 1
    return confidence / num_tokens

def generate(
    tokenizer,
    model,
    reward_model: RewardModel,
    question: str,
    num_thought_tokens: int = 10,
    lr: float = 0.05,
    sigma: float = 0.1,
    sigma_decay: float = 0.99,
    max_rl_steps: int = 10,
    reward_threshold: float = -1,
    max_new_tokens: int = 4096,
    use_auto_grad: bool = True,
    disable_conf_reward: bool = False,
    disable_best_reward: bool = False,
    data_name: str = None,
    model_name: str = None,
    verbose: int = 1,
    top_k: int = 10,
    **kwargs,
):
    model.eval()

    inputs, thought_idx = build_inputs(
        tokenizer=tokenizer,
        num_thought_tokens=num_thought_tokens,
        prompt=question,
        data_name=data_name,
        model_name=model_name,
    )

    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
    inputs['inputs_embeds'] = inputs_embeds
    inputs.pop('input_ids')

    device = inputs_embeds.device
    # outputs = model(**inputs, output_hidden_states=True)
    # thought_hidden_states = outputs['hidden_states'][-1][0, thought_idx[0]:thought_idx[1]].clone()
    if not disable_conf_reward and use_auto_grad:
        thought_hidden_states = torch.nn.Parameter(
            inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone().detach().requires_grad_(True)
        )
        optimizer = torch.optim.Adam([thought_hidden_states], lr=lr, maximize=True)
    else:
        thought_hidden_states = inputs_embeds[0, thought_idx[0]:thought_idx[1]].clone()

    best_reward = 0.0
    best_reward_step = 0
    best_thought_hidden_states = thought_hidden_states.clone()
    for i in range(max_rl_steps):
        if not disable_conf_reward and use_auto_grad:
            optimizer.zero_grad()
        epsilon = torch.normal(mean=0.0, std=sigma, size=thought_hidden_states.shape).to(device)
        thought_hidden_states_cand = thought_hidden_states + epsilon
        if disable_conf_reward:
            with torch.no_grad():
                reward = reward_model.get_reward(
                    question=question,
                    specil_tokens_embeds=thought_hidden_states_cand,
                )
        else:
            if use_auto_grad:
                reward = get_confidence(
                    model=model,
                    inputs=inputs,
                    thought_idx=thought_idx,
                    thought_hidden_states=thought_hidden_states_cand,
                    k=top_k,
                )
                reward.requires_grad_(True)
                reward.backward(retain_graph=True)
            else:
                with torch.no_grad():
                    reward = get_confidence(
                        model=model,
                        inputs=inputs,
                        thought_idx=thought_idx,
                        thought_hidden_states=thought_hidden_states_cand,
                        k=top_k,
                    )
        grad_ascent = None
        if not disable_conf_reward and use_auto_grad:
            optimizer.step()
        else:
            grad_ascent = lr * reward * epsilon / sigma**2
            thought_hidden_states += grad_ascent
        sigma *= sigma_decay

        if verbose:
            logger.info(f'>>> Step {i} reward = {reward}')

        del epsilon, thought_hidden_states_cand
        torch.cuda.empty_cache()

        if reward > best_reward:
            best_reward = reward
            best_reward_step = i
            best_thought_hidden_states = thought_hidden_states.clone()
        if reward_threshold > 0 and reward >= reward_threshold:
            break

    if disable_best_reward:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = thought_hidden_states
    else:
        inputs_embeds[0, thought_idx[0]:thought_idx[1]] = best_thought_hidden_states
    inputs['inputs_embeds'] = inputs_embeds

    kwargs.update(dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        num_beams=1,
    ))
    outputs = model.generate(**inputs, **kwargs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, best_reward, best_reward_step
