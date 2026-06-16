import dataclasses
import sys
from logging import INFO, WARNING, getLogger

import torch
from src.simulation.langevin2d import (
    InfoThermodynLangevin2D,
    calc_evolution_mu_and_sigma,
    get_steady_cov_mat,
)
from src.simulation.seasonal_sro import SeasonalSROConfig

if "ipykernel" in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

logger = getLogger()


def get_langevin_without_forcing_for_seasonal_SRO(
    cfg: SeasonalSROConfig,
    t_min: float,
    t_max: float,
    n_timesteps: int,
    dtype: torch.dtype = torch.float64,
    mu0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, InfoThermodynLangevin2D]:

    assert t_max > t_min
    assert isinstance(n_timesteps, int) and n_timesteps > 0

    dt = (t_max - t_min) / n_timesteps
    ts = torch.linspace(t_min, t_max, n_timesteps + 1, dtype=dtype)

    As, Bs = cfg.mat_A(ts, dtype), cfg.mat_B(ts, dtype)
    assert As.shape == Bs.shape == (n_timesteps + 1, 2, 2)
    cov0 = get_steady_cov_mat(A=As[0], B=Bs[0])
    assert cov0.shape == (2, 2)

    if mu0 is None:
        mu0 = torch.zeros(2, dtype=dtype)
    else:
        assert mu0.shape == (2,)
        mu0 = mu0.to(dtype)
        logger.info(f"Initial mean mu0: {mu0}")

    fs = torch.zeros(n_timesteps + 1, 2, dtype=dtype)

    mu, cov = calc_evolution_mu_and_sigma(
        dt=dt, fs=fs, As=As, Bs=Bs, mu0=mu0, sigma0=cov0
    )

    logger.setLevel(WARNING)
    langevin = InfoThermodynLangevin2D(dt=dt, mu=mu, sigma=cov, As=As, Bs=Bs, fs=fs)
    logger.setLevel(INFO)

    return ts, langevin


@dataclasses.dataclass()
class InformationThermodynamicsResult:
    itrX: torch.Tensor
    iflX: torch.Tensor
    qX: torch.Tensor
    sX: torch.Tensor

    itrY: torch.Tensor
    iflY: torch.Tensor
    qY: torch.Tensor
    sY: torch.Tensor

    s_tot: torch.Tensor
    s_x_cond_y: torch.Tensor

    x_ave: torch.Tensor
    y_ave: torch.Tensor
    x_bar: torch.Tensor

    bound_info: torch.Tensor
    bound_conv: torch.Tensor
    chi: torch.Tensor
    var_bound_info: torch.Tensor
    var_bound_conv: torch.Tensor
    var: torch.Tensor

    fac: torch.Tensor
    offset: torch.Tensor

    dvar_dt: torch.Tensor
    dvar_dt_sq: torch.Tensor
    bound_dvar_dt_sq: torch.Tensor
    bound_var: torch.Tensor

    tur_current: torch.Tensor
    tur_pt: torch.Tensor

    def __init__(self, langevin: InfoThermodynLangevin2D):
        self.itrX = langevin.Itr_y_to_x()
        self.iflX = langevin.I_y_to_x()
        self.qX = langevin.Qx_over_Tx()
        self.sX = langevin.dSx_dt()

        # self.itrY = langevin.Itr_x_to_y()
        # self.iflY = langevin.I_x_to_y()
        # self.qY = langevin.Qy_over_Ty()
        # self.sY = langevin.dSy_dt()

        # self.s_tot = langevin.S_tot()
        # self.s_x_cond_y = langevin.S_x_cond_y()

        self.x_ave = langevin.x_ave()
        self.y_ave = langevin.y_ave()
        # self.x_bar = langevin.xs_equilibrium()

        # self.bound_info = self.itrY + self.s_x_cond_y
        # self.bound_conv = self.qY + self.s_tot
        # self.chi = 1.0 - (self.qX + self.bound_info) / (self.qX + self.bound_conv)

        # self.fac = (langevin.rx[:-1] ** 2) / langevin.Tx[:-1]
        # self.offset = langevin.Tx[:-1] / langevin.rx[:-1]

        # self.var_bound_info = -self.bound_info / self.fac + self.offset
        # self.var_bound_conv = -self.bound_conv / self.fac + self.offset
        # self.var = self.qX / self.fac + self.offset

        # self.dvar_dt = langevin.dx_dt_cdot_x()
        self.dvar_dt_sq = langevin.dx_dt_cdot_x() ** 2

        s = self.sX + self.qX - self.iflX
        var = langevin.x2_ave()[:-1]
        T = langevin.Tx[:-1]
        self.bound_dvar_dt_sq = s * var * T

        # self.bound_var = self.dvar_dt_sq / (T * s)

        self.tur_current = self.dvar_dt_sq / (T * var)
        self.tur_pt = self.sX + self.qX - self.iflX


def calc_diff_bound(
    t_min: float, t_max: float, nt_year: int, config: SeasonalSROConfig
) -> tuple[float, float, float]:

    n_timesteps = int((t_max - t_min) * nt_year)

    mu0 = torch.tensor([0.0, 10.0], dtype=torch.float64)
    _, langevin = get_langevin_without_forcing_for_seasonal_SRO(
        cfg=config, t_min=t_min, t_max=t_max, n_timesteps=n_timesteps, mu0=mu0
    )
    result = InformationThermodynamicsResult(langevin)

    data = torch.sqrt(result.tur_pt[-nt_year:])

    vmax, vmin = data.max().item(), data.min().item()
    diff_bound = vmax - vmin

    return diff_bound, vmax, vmin
