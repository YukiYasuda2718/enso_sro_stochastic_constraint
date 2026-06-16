"""
$$
d \mathbf{x} = -A \mathbf{x} dt + B d\mathbf{W}
$$
"""

import copy
from logging import getLogger

import torch

logger = getLogger()


def get_steady_cov_mat(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    \begin{equation}
        \Sigma=\frac{({\rm det} A) B B^T+(A-{\rm tr} A) B B^T(A^T-{\rm tr} A)}{2 {\rm tr} A {\rm det} A}
    \end{equation}
    """
    assert A.shape == B.shape == (2, 2), "only 2D is supported."

    det = torch.linalg.det(A)
    tr = torch.trace(A)
    C = A - tr * torch.eye(2, dtype=A.dtype)
    D = torch.mm(B, B.T)

    x = det * D + torch.mm(torch.mm(C, D), C.T)
    y = 2.0 * tr * det

    return x / y


def calc_evolution_mu_and_sigma(
    *,
    dt: float,
    fs: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    mu0: torch.Tensor,
    sigma0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    \begin{align}
        \frac{d \mu}{d t}&=-A \mu(t)+f(t) \\
        \frac{d \Sigma}{d t}&=-A \Sigma-\Sigma A^{\rm T}+B B^{\rm T}
    \end{align}
    """
    n_t = fs.shape[0]  # num of times
    dim = As.shape[1]

    assert fs.shape == (n_t, dim)
    assert mu0.shape == (dim,)
    assert sigma0.shape == (dim, dim)
    assert As.shape == Bs.shape == (n_t, dim, dim)

    mu = torch.zeros((n_t, dim), dtype=As.dtype)
    sigma = torch.zeros((n_t, dim, dim), dtype=As.dtype)

    mu[0] = copy.deepcopy(mu0)
    sigma[0] = copy.deepcopy(sigma0)

    for i in range(1, n_t, 1):
        mu[i] = mu[i - 1] + dt * (torch.mv(-As[i - 1], mu[i - 1]) + fs[i - 1])
        E = torch.mm(-As[i - 1], sigma[i - 1])
        D = torch.mm(Bs[i - 1], Bs[i - 1].T)
        sigma[i] = sigma[i - 1] + dt * (E + E.T + D)
    return mu, sigma


class InfoThermodynLangevin2D:
    def __init__(
        self,
        *,
        dt: float,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        fs: torch.Tensor,
    ):
        assert mu.ndim == 2
        n_t, dim = mu.shape
        assert dim == 2, "Only 2D is supported."
        assert sigma.shape == (n_t, dim, dim)
        assert As.shape == (n_t, dim, dim)
        assert Bs.shape == (n_t, dim, dim)
        assert fs.shape == (n_t, dim)

        self.dt = dt
        self.mu = copy.deepcopy(mu)
        self.sigma = copy.deepcopy(sigma)
        self.det = torch.linalg.det(self.sigma)
        self.As = copy.deepcopy(As)
        self.Bs = copy.deepcopy(Bs)
        self.fs = copy.deepcopy(fs)

        self.sigma_x_cond_y = (
            self.sigma[:, 0, 0]
            - (self.sigma[:, 0, 1] / self.sigma[:, 1, 1]) * self.sigma[:, 1, 0]
        )

        self.Tx = (Bs[:, 0, 0] ** 2) / 2.0
        self.Ty = (Bs[:, 1, 1] ** 2) / 2.0

        self.rx = As[:, 0, 0]
        self.ry = As[:, 1, 1]
        logger.info(f"rx = {torch.mean(self.rx)}, ry = {torch.mean(self.ry)}")

        self.a = -As[:, 0, 1]
        self.b = -As[:, 1, 0]
        logger.info(f"a = {torch.mean(self.a)}, b = {torch.mean(self.b)}")

        self.a11 = -self.rx  #  -As[:, 0, 0]
        self.a12 = self.a  #  -As[:, 0, 1]
        self.a21 = self.b  #  -As[:, 1, 0]
        self.a22 = -self.ry  #  -As[:, 1, 1]
        logger.info(f"a11 = {torch.mean(self.a11)}")
        logger.info(f"a12 = {torch.mean(self.a12)}")
        logger.info(f"a21 = {torch.mean(self.a21)}")
        logger.info(f"a22 = {torch.mean(self.a22)}")

    def x_ave(self) -> torch.Tensor:
        return self.mu[:, 0]

    def y_ave(self) -> torch.Tensor:
        return self.mu[:, 1]

    def x2_ave(self) -> torch.Tensor:
        return self.sigma[:, 0, 0] + self.mu[:, 0] ** 2

    def y2_ave(self) -> torch.Tensor:
        return self.sigma[:, 1, 1] + self.mu[:, 1] ** 2

    def xy_ave(self) -> torch.Tensor:
        return self.sigma[:, 0, 1] + self.mu[:, 0] * self.mu[:, 1]

    def S_tot(self) -> torch.Tensor:
        inv = torch.linalg.inv(self.sigma)
        s_tot = (self.sigma[1:] - self.sigma[:-1]) / self.dt
        s_tot = torch.bmm(s_tot, inv[:-1])
        return 0.5 * torch.einsum("kii->k", s_tot)

    def S_x_cond_y(self) -> torch.Tensor:
        s = (self.sigma_x_cond_y[1:] - self.sigma_x_cond_y[:-1]) / self.dt
        s = s / self.sigma_x_cond_y[:-1]
        return 0.5 * s

    def Itr_x_to_y(self) -> torch.Tensor:
        ret = (self.b**2) * self.det / (4 * self.Ty * self.sigma[:, 1, 1])
        return ret[:-1]

    def Itr_y_to_x(self) -> torch.Tensor:
        ret = (self.a**2) * self.det / (4 * self.Tx * self.sigma[:, 0, 0])
        return ret[:-1]

    def Qx_over_Tx(self) -> torch.Tensor:
        x = (self.rx**2) * self.x2_ave()
        x = x - 2.0 * self.rx * self.a * self.xy_ave()
        x = x + (self.a**2) * self.y2_ave()
        x = x + self.fs[:, 0] ** 2

        y = -self.rx * self.x_ave() + self.a * self.y_ave()
        x = x + 2.0 * self.fs[:, 0] * y

        q = x / self.Tx - self.rx

        return q[:-1]

    def Qy_over_Ty(self) -> torch.Tensor:
        x = (self.ry**2) * self.y2_ave()
        x = x - 2.0 * self.ry * self.b * self.xy_ave()
        x = x + (self.b**2) * self.x2_ave()
        x = x + self.fs[:, 1] ** 2

        y = -self.ry * self.y_ave() + self.b * self.x_ave()
        x = x + 2 * self.fs[:, 1] * y

        q = x / self.Ty - self.ry

        return q[:-1]

    def dSx_dt(self) -> torch.Tensor:
        x = self.a11
        x = x + self.a12 * self.sigma[:, 0, 1] / self.sigma[:, 0, 0]
        x = x + self.Tx / self.sigma[:, 0, 0]
        return x[:-1]

    def dSy_dt(self) -> torch.Tensor:
        x = self.a22
        x = x + self.a21 * self.sigma[:, 1, 0] / self.sigma[:, 1, 1]
        x = x + self.Ty / self.sigma[:, 1, 1]
        return x[:-1]

    def I_y_to_x(self) -> torch.Tensor:
        x = self.a12 * self.sigma[:, 0, 1] / self.sigma[:, 0, 0]
        x = x + self.Tx / self.sigma[:, 0, 0]
        x = x - self.Tx * self.sigma[:, 1, 1] / self.det
        return x[:-1]

    def I_x_to_y(self) -> torch.Tensor:
        x = self.a21 * self.sigma[:, 1, 0] / self.sigma[:, 1, 1]
        x = x + self.Ty / self.sigma[:, 1, 1]
        x = x - self.Ty * self.sigma[:, 0, 0] / self.det
        return x[:-1]

    def Varx_from_forcing(self) -> torch.Tensor:
        x = self.x2_ave()
        x = x + (self.fs[:, 0] / self.rx) ** 2
        ret = x - 2.0 * self.fs[:, 0] * self.x_ave() / self.rx
        return ret[:-1]

    def effect_of_a(
        self, is_split: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        y = self.y2_ave() * ((self.a / self.rx) ** 2)
        z = -2.0 * self.xy_ave() * self.a / self.rx
        w = 2.0 * self.y_ave() * self.fs[:, 0] * self.a / (self.rx**2)

        if is_split:
            return y[:-1], z[:-1], w[:-1]
        return y[:-1] + z[:-1] + w[:-1]

    def xs_equilibrium(self):
        x = (self.a * self.y_ave() + self.fs[:, 0]) / self.rx
        return x[:-1]

    def dx_dt_cdot_x(self):
        x = -self.rx * self.x2_ave()
        x = x + self.a * self.xy_ave()
        x = x + self.Tx
        return x[:-1]
