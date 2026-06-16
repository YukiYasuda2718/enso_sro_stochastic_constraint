import copy
import dataclasses
import gc
import math
import os
import sys
from logging import getLogger
from typing import Literal

import numpy as np
import torch
from src.utils.io_pickles import write_pickle
from src.utils.random_seed_helper import set_all_seeds

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


@dataclasses.dataclass()
class SeasonalSROConfig:
    """
    Ref: Vialard et al. (2025), "The El Niño Southern Oscillation (ENSO) Recharge Oscillator Conceptual Model: Achievements and Future Prospects"
    We assume the following SDE: dx = - A x dt + B dW, where x = (T, h)
    """

    R0: float = -1.13  # Bjerknes feedback (time mean), unit: 1/yr
    Ra: float = 1.13 * 2.0  # Bjerknes feedback (amplitude), unit: 1/yr
    phi: float = 2 * math.pi / 24.0  # Bjerknes feedback (phase), unit: radian
    F1: float = 0.19  # Delayed oceanic feedback efficiency, unit: K/m/yr
    epsilon: float = 0.38  # Basin adjustment, unit: 1/yr
    F2: float = 17.58  # Recharge/discharge efficiency, unit: m/K/yr
    sigma_T: float = 2.22  # Stochastic forcing amplitude for T, unit: K/yr
    sigma_h: float = 14.37  # Stochastic forcing amplitude for h, unit: m/yr
    value_set_name: Literal[
        "KA21WeakNoise", "KA21StrongNoise", "KA21", "V25", "H26"
    ] = "V25"

    def __post_init__(self):
        if self.value_set_name == "KA21" or self.value_set_name == "KA21WeakNoise":
            self.set_KA21_weak_noise_params()
            logger.info("Using KA21 weak noise parameters.")
        elif self.value_set_name == "KA21StrongNoise":
            self.set_KA21_strong_noise_params()
            logger.info("Using KA21 strong noise parameters.")
        elif self.value_set_name == "V25":
            logger.info("Using V25 parameters.")
            pass
        elif self.value_set_name == "H26":
            self.set_H26_params()
            logger.info("Using H26 parameters.")
        else:
            raise ValueError(
                f"Unknown value_set_name: {self.value_set_name}. Supported: 'KA21WeakNoise', 'KA21StrongNoise', 'V25', 'H26'."
            )

    def get_bwj_real_part(self) -> float:
        return (self.R0 - self.epsilon) / 2.0

    def get_bwj_imaginary_part(self) -> float:
        return math.sqrt(self.F1 * self.F2 - ((self.R0 + self.epsilon) ** 2) / 4.0)

    def get_Rt(self, years: torch.Tensor) -> torch.Tensor:
        return self.R0 - self.Ra * torch.sin(2 * math.pi * years + self.phi)

    def mat_A(
        self, years: torch.Tensor, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        A = torch.tensor([[-self.R0, -self.F1], [self.F2, self.epsilon]], dtype=dtype)
        n_timesteps = len(years)
        As = A[None, ...].repeat(n_timesteps, 1, 1)
        As[:, 0, 0] = -self.get_Rt(years)
        return As

    def mat_B(
        self, years: torch.Tensor, dtype: torch.dtype = torch.float64
    ) -> torch.Tensor:
        B = torch.tensor([[self.sigma_T, 0], [0, self.sigma_h]], dtype=dtype)
        n_timesteps = len(years)
        shape = (n_timesteps, 2, 2)
        return torch.broadcast_to(B, shape)

    def set_KA21_weak_noise_params(self):
        # 1/month --> 1/year
        self.R0 = -0.0748 * 12
        self.Ra = abs(self.R0) * 2.0
        self.F1 = 0.0202 * 12
        self.F2 = 0.9138 * 12
        self.epsilon = 0.0316 * 12
        self.sigma_T = 0.23 * math.sqrt(12)
        self.sigma_h = 1.68 * math.sqrt(12)

    def set_KA21_strong_noise_params(self):
        # 1/month --> 1/year
        self.R0 = -0.0748 * 12
        self.Ra = abs(self.R0) * 2.0
        self.F1 = 0.0202 * 12
        self.F2 = 0.9138 * 12
        self.epsilon = 0.0316 * 12
        self.sigma_T = 0.23 * 12
        self.sigma_h = 1.68 * 12

    def set_H26_params(self):
        # 1/month --> 1/year
        self.R0 = 0.03 * 12
        self.Ra = 0.16 * 12
        self.F1 = 0.015 * 12
        self.F2 = 1.45 * 12
        self.epsilon = 0.13 * 12
        self.sigma_T = 0.176 * math.sqrt(12)
        self.sigma_h = 1.60 * math.sqrt(12)


def simulate_SeasonalSRO(
    x0: torch.Tensor,
    t_min: float,
    t_max: float,
    n_timesteps: int,
    n_batches: int,
    config: SeasonalSROConfig,
    seed: int | None = 42,
    rng: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:

    assert t_max > t_min and n_timesteps > 0 and n_batches > 0
    assert x0.shape == (n_batches, 2)

    dt = (t_max - t_min) / n_timesteps
    ts = torch.linspace(t_min, t_max, n_timesteps + 1, dtype=dtype)

    mat_A = config.mat_A(ts, dtype)
    mat_B = config.mat_B(ts, dtype)
    assert mat_A.shape == mat_B.shape == (n_timesteps + 1, 2, 2)

    xs: list[torch.Tensor] = [x0.detach().clone().to(dtype)]
    x = x0.detach().clone().to(dtype)

    if rng is None:
        assert seed is not None
        set_all_seeds(seed)
        dWs = math.sqrt(dt) * torch.randn(n_batches, n_timesteps, 2, dtype=dtype)
    else:
        dWs = math.sqrt(dt) * torch.randn(
            n_batches, n_timesteps, 2, dtype=dtype, generator=rng
        )

    if show_progress:
        iterator = tqdm(range(n_timesteps))
    else:
        iterator = range(n_timesteps)

    for it in iterator:
        x = (
            x
            + dt * torch.einsum("ij,bj->bi", -mat_A[it], x)
            + torch.einsum("ij,bj->bi", mat_B[it], dWs[:, it])
        )
        xs.append(x.detach().clone())

    # stack along time. Shape: (n_batches, n_timesteps+1, 2)
    stacked = torch.stack(xs, dim=1)

    return ts, stacked


def simulate_SeasonalSRO_and_write_out_results(
    amp_Ra: float,
    base_seed: int,
    default_config: SeasonalSROConfig,
    x0: torch.Tensor,
    t_min: float,
    t_max: float,
    n_timesteps: int,
    n_batches: int,
    out_file_path: str,
    dtype_save: str = "float32",
    worker_index: int = 0,
):
    if os.path.exists(out_file_path):
        return {
            "index": worker_index,
            "amp_Ra": amp_Ra,
            "path": out_file_path,
            "status": "skipped-exists",
        }

    assert t_max > t_min and n_timesteps > 0 and n_batches > 0
    assert x0.shape == (n_batches, 2)

    os.makedirs(os.path.dirname(out_file_path), exist_ok=True)

    ss = np.random.SeedSequence([int(base_seed), int(worker_index)])
    st = ss.generate_state(2, dtype=np.uint32)
    child_seed = (int(st[0]) << 32) | int(st[1])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(child_seed)

    config = copy.deepcopy(default_config)
    config.Ra = abs(config.R0) * amp_Ra

    with torch.no_grad():
        ts, xs = simulate_SeasonalSRO(
            x0=x0,
            t_min=t_min,
            t_max=t_max,
            n_timesteps=n_timesteps,
            n_batches=n_batches,
            config=config,
            seed=None,
            rng=gen,
            dtype=torch.float64,
            show_progress=False,
        )

    ts_np = ts.detach().cpu().numpy().astype(dtype_save)
    xs_np = xs.detach().cpu().numpy().astype(dtype_save)

    payload = dict(
        amp_Ra=float(amp_Ra),
        seed=int(child_seed),
        x0=x0.detach().cpu().numpy().astype(dtype_save),
        t_min=float(t_min),
        t_max=float(t_max),
        n_timesteps=int(n_timesteps),
        n_batches=int(n_batches),
        config=config,
        ts=ts_np,
        xs=xs_np,
        index=int(worker_index),
    )
    write_pickle(payload, out_file_path)

    del xs, ts, ts_np, xs_np, payload
    gc.collect()

    return {
        "index": worker_index,
        "amp_Ra": amp_Ra,
        "path": out_file_path,
        "status": "ok",
    }
