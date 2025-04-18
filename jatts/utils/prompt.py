import numpy as np

def prepare_prompt(prefix_mode:int, prompt, max_prompt_length:int):
    """
    Prepare prompt.

    Args:
        prefix_mode (int): Prefix mode.
        prompt: Prompt feature. Shape: [t, q]
        max_prompt_length (int): Maximum prompt length.
    
    """

    # mode 1: randomly crop max_prompt_length frames
    if prefix_mode == 1:
        if prompt.shape[0] > max_prompt_length:
            start = np.random.randint(
                0, prompt.shape[0] - max_prompt_length
            )
            prompt = prompt[start : start + max_prompt_length]
    else:
        raise ValueError(f"Unsupported prefix mode: {prefix_mode}")

    return prompt