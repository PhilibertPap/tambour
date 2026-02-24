from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


class ConstitutiveLaw:
    """Minimal constitutive interface exposing nominal stresses P_rr and P_zz."""

    def P_rr(self, lambda_r: float, lambda_z: float) -> float:
        raise NotImplementedError

    def P_zz(self, lambda_r: float, lambda_z: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class _KinematicsInvariants:
    """Convenience container for invariants used repeatedly in stress formulas."""

    J: float
    I1: float
    J_to_minus_two_thirds: float


class NeoHookeanReducedI1(ConstitutiveLaw):
    r"""
    Compressible neo-Hookean model (reduced I1 form) in the axisymmetric homogeneous setting.

    The implemented nominal stress follows the split used in the report:
      P = 2 mu J^{-2/3}(F - (I1/3) F^{-T}) + (kappa/2)(J^2-1) F^{-T}

    with:
      F = diag(lambda_r, lambda_r, lambda_z),
      J = lambda_r^2 lambda_z,
      I1 = 2 lambda_r^2 + lambda_z^2.
    """

    def __init__(self, mu: float, K: float):
        self.mu = float(mu)
        self.K = float(K)  # volumetric modulus kappa

    @staticmethod
    def _compute_invariants(lambda_r: float, lambda_z: float) -> _KinematicsInvariants:
        J = lambda_r * lambda_r * lambda_z
        I1 = 2.0 * lambda_r * lambda_r + lambda_z * lambda_z
        return _KinematicsInvariants(J=J, I1=I1, J_to_minus_two_thirds=J ** (-2.0 / 3.0))

    def _isochoric_part(self, lambda_i: float, invariants: _KinematicsInvariants) -> float:
        return (
            2.0
            * self.mu
            * invariants.J_to_minus_two_thirds
            * (lambda_i - invariants.I1 / (3.0 * lambda_i))
        )

    def _volumetric_part(self, lambda_i: float, invariants: _KinematicsInvariants) -> float:
        volumetric_prefactor = 0.5 * self.K * (invariants.J * invariants.J - 1.0)
        return volumetric_prefactor / lambda_i

    def P_rr(self, lambda_r: float, lambda_z: float) -> float:
        invariants = self._compute_invariants(lambda_r, lambda_z)
        return self._isochoric_part(lambda_r, invariants) + self._volumetric_part(lambda_r, invariants)

    def P_zz(self, lambda_r: float, lambda_z: float) -> float:
        invariants = self._compute_invariants(lambda_r, lambda_z)
        return self._isochoric_part(lambda_z, invariants) + self._volumetric_part(lambda_z, invariants)


def mu_K_from_E_nu(E: float, nu: float) -> Tuple[float, float]:
    """
    Isotropic linear-elastic parameter conversion:
      mu    = E / (2 (1 + nu))
      kappa = E / (3 (1 - 2 nu))
    """
    E = float(E)
    nu = float(nu)
    if E <= 0.0:
        raise ValueError("E doit etre > 0.")
    if not (-1.0 < nu < 0.5):
        raise ValueError("nu doit etre dans (-1, 0.5) pour un modele isotrope stable.")

    mu = E / (2.0 * (1.0 + nu))
    K = E / (3.0 * (1.0 - 2.0 * nu))
    return mu, K


def E_nu_from_mu_K(mu: float, K: float) -> Tuple[float, float]:
    """
    Inverse isotropic linear-elastic conversion:
      E  = 9 K mu / (3 K + mu)
      nu = (3 K - 2 mu) / (2 (3 K + mu))
    """
    mu = float(mu)
    K = float(K)
    if mu <= 0.0 or K <= 0.0:
        raise ValueError("mu et K doivent etre > 0.")

    denominator = 3.0 * K + mu
    E = 9.0 * K * mu / denominator
    nu = (3.0 * K - 2.0 * mu) / (2.0 * denominator)
    return E, nu


def drum_material_preset(name: str) -> Tuple[float, float]:
    """
    Typical (E, nu) presets used for order-of-magnitude studies.

    Available presets:
      - mylar_92
      - steel_304
    """
    key = name.strip().lower()
    if key == "mylar_92":
        return 3.5e9, 0.38
    if key == "steel_304":
        return 193e9, 0.29
    raise ValueError("Preset inconnu. Choix: mylar_92, steel_304.")
