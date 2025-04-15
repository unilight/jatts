from torch.optim.lr_scheduler import SequentialLR, LinearLR


def E2TTSSequentialLR(
    optimizer,
    warmup_steps,
    decay_steps,
    warmup_start_factor,
    warmup_end_factor,
    decay_start_factor,
    decay_end_factor,
):

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=warmup_end_factor,
        total_iters=warmup_steps,
    )
    decay_scheduler = LinearLR(
        optimizer,
        start_factor=decay_start_factor,
        end_factor=decay_end_factor,
        total_iters=decay_steps,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps],
    )
