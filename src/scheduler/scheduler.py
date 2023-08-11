def update_lrate(d_model: int, step_num: int, warmup_steps: int) -> float:
    lrate = d_model**-0.5 * min(step_num**-0.5, step_num * warmup_steps**-1.5)
    return lrate
