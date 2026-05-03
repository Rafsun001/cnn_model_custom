import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    def __init__(self, drop_path_rate=0.0):
        super().__init__()
        if drop_path_rate < 0.0 or drop_path_rate >= 1.0:
            raise ValueError("drop_path_rate must be in the range [0.0, 1.0).")
        self.drop_path_rate = float(drop_path_rate)

    def forward(self, x):
        if self.drop_path_rate == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x * random_tensor / keep_prob


def make_drop_path_rates(total_blocks, final_drop_path_rate):
    if final_drop_path_rate < 0.0 or final_drop_path_rate >= 1.0:
        raise ValueError("drop_path_rate must be in the range [0.0, 1.0).")
    if total_blocks <= 1:
        return [float(final_drop_path_rate)]
    return torch.linspace(0.0, final_drop_path_rate, total_blocks).tolist()


def get_stage_drop_path_rates(drop_path_rates, num_blocks):
    if drop_path_rates is None:
        return [0.0] * num_blocks
    if len(drop_path_rates) != num_blocks:
        raise ValueError(f"Expected {num_blocks} drop path rates, got {len(drop_path_rates)}.")
    return drop_path_rates
