from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from models import NeoHookeanReducedI1, drum_material_preset, mu_K_from_E_nu
from solvers import ContinuationSettings, NewtonSettings, continuation_in_T


# ============================================================
# PARAMETRES A MODIFIER DIRECTEMENT
# ============================================================

# --- Materiau ---
MATERIAL_PRESET = "peau"   # ex: "peau", "mylar_92", "steel_304"
# E = ...
# NU = ...

# MU = ...
# KBULK = ...

# --- Charge ---
LOAD_PRESET = "drum_classic_tendu"
# Choix possibles :
# "drum_classic_relax"
# "drum_classic_tendu"
# "drum_classic_tres_tendu"

# T_MAX = ...
# DT0 = ...
# DT_MAX = ...

# --- Geometrie et masse volumique ---
RADIUS = 0.17
RHO0 = 1200.0

# --- Modes ---
MODES = ["01", "11", "21", "02", "31", "12"]

# --- Exports ---
EXPORT_CSV = None
# EXPORT_CSV = "results/frequencies.csv"

SAVE_PREFIX = None
# SAVE_PREFIX = "results/coupled_peau"

SHOW_FIGURES = True


# ============================================================
# CONSTANTES
# ============================================================

MODE_ROOTS: Dict[str, float] = {
    "01": 2.4048255577,
    "11": 3.8317059702,
    "21": 5.1356223018,
    "02": 5.5200781103,
    "31": 6.3801618952,
    "12": 7.0155866698,
}


@dataclass
class PrestrainCurve:
    T0: np.ndarray
    lambda_R: np.ndarray
    lambda_Z: np.ndarray
    J1: np.ndarray


@dataclass
class FrequencyCurve:
    mode: str
    alpha_nm: float
    omega: np.ndarray
    frequency_hz: np.ndarray


def load_traction_preset(name: str, mu: float) -> Tuple[float, float, float]:
    key = name.strip().lower()
    if key == "drum_classic_relax":
        return 0.10 * mu, 0.005 * mu, 0.02 * mu
    if key == "drum_classic_tendu":
        return 0.35 * mu, 0.015 * mu, 0.05 * mu
    if key == "drum_classic_tres_tendu":
        return 0.70 * mu, 0.03 * mu, 0.10 * mu
    raise ValueError(
        "Preset de charge inconnu. Choix: drum_classic_relax, drum_classic_tendu, drum_classic_tres_tendu."
    )


def validate_modes(modes: Sequence[str]) -> List[str]:
    checked_modes: List[str] = []
    for mode in modes:
        if mode not in MODE_ROOTS:
            raise ValueError(
                "Mode inconnu '{}'. Modes disponibles: {}".format(mode, ", ".join(sorted(MODE_ROOTS)))
            )
        checked_modes.append(mode)
    if not checked_modes:
        raise ValueError("Aucun mode valide n'a ete fourni.")
    return checked_modes


def pick_material_parameters() -> Tuple[float, float]:
    """
    Priorite :
    1) preset materiau
    2) E, nu
    3) mu, kappa
    """
    if "MATERIAL_PRESET" in globals() and MATERIAL_PRESET is not None:
        E, nu = drum_material_preset(MATERIAL_PRESET)
        return mu_K_from_E_nu(E, nu)

    if "E" in globals() and "NU" in globals():
        return mu_K_from_E_nu(E, NU)

    if "MU" in globals() and "KBULK" in globals():
        return float(MU), float(KBULK)

    raise ValueError("Aucun parametre materiau valide n'a ete fourni.")


def solve_prestrain_continuation(
    mu: float, kappa: float, T_max: float, dT0: float, dT_max: float
) -> PrestrainCurve:
    constitutive_law = NeoHookeanReducedI1(mu=mu, K=kappa)

    def make_residual(T0):
        def residual(log_lambda):
            lambda_R = float(np.exp(log_lambda[0]))
            lambda_Z = float(np.exp(log_lambda[1]))
            return np.array(
                [
                    constitutive_law.P_rr(lambda_R, lambda_Z) - T0,
                    constitutive_law.P_zz(lambda_R, lambda_Z),
                ],
                dtype=float,
            )
        return residual

    continuation_solution = continuation_in_T(
        make_residual=make_residual,
        stN=NewtonSettings(),
        stC=ContinuationSettings(T_max=T_max, dT0=dT0, dT_max=dT_max),
        x_init=np.array([0.0, 0.0]),
    )

    T0_values = np.array([step[0] for step in continuation_solution], dtype=float)
    log_lambdas = np.array([step[1] for step in continuation_solution], dtype=float)

    lambda_R = np.exp(log_lambdas[:, 0])
    lambda_Z = np.exp(log_lambdas[:, 1])
    J1 = lambda_R**2 * lambda_Z

    return PrestrainCurve(
        T0=T0_values,
        lambda_R=lambda_R,
        lambda_Z=lambda_Z,
        J1=J1,
    )


def compute_effective_stiffness(curve: PrestrainCurve, mu: float) -> np.ndarray:
    return (
        curve.T0 / curve.lambda_R
        + mu * (curve.J1 ** (-2.0 / 3.0)) * (curve.lambda_Z ** 2) / (curve.lambda_R ** 2)
    )


def compute_frequency_curves(
    modes: Sequence[str], radius_A: float, rho0: float, K_eff: np.ndarray
) -> List[FrequencyCurve]:
    curves: List[FrequencyCurve] = []
    wave_speed = np.sqrt(K_eff / rho0)

    for mode in modes:
        alpha_nm = MODE_ROOTS[mode]
        omega = wave_speed * (alpha_nm / radius_A)
        curves.append(
            FrequencyCurve(
                mode=mode,
                alpha_nm=alpha_nm,
                omega=omega,
                frequency_hz=omega / (2.0 * np.pi),
            )
        )
    return curves


def build_summary_lines(
    curve: PrestrainCurve, mu: float, K_eff: np.ndarray, frequency_curves: Sequence[FrequencyCurve]
) -> List[str]:
    indices = [0, len(curve.T0) // 2, len(curve.T0) - 1]
    labels = ["debut", "milieu", "fin"]

    lines: List[str] = []
    for label, i in zip(labels, indices):
        chunks = [
            "{}: T0={:.6e} Pa".format(label, curve.T0[i]),
            "T0/mu={:.6e}".format(curve.T0[i] / mu),
            "lambda_R={:.8f}".format(curve.lambda_R[i]),
            "lambda_Z={:.8f}".format(curve.lambda_Z[i]),
            "J1-1={:.3e}".format(curve.J1[i] - 1.0),
            "Keff={:.6e} Pa".format(K_eff[i]),
        ]
        for mode_curve in frequency_curves:
            chunks.append("f{}={:.3f} Hz".format(mode_curve.mode, mode_curve.frequency_hz[i]))
        lines.append(", ".join(chunks))

    return lines


def export_frequency_csv(
    path: str,
    curve: PrestrainCurve,
    mu: float,
    K_eff: np.ndarray,
    radius_A: float,
    rho0: float,
    frequency_curves: Sequence[FrequencyCurve],
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "T0_Pa",
        "T0_over_mu",
        "lambda_R",
        "lambda_Z",
        "J1",
        "J1_minus_1",
        "Keff_Pa",
        "Keff_over_mu",
        "radius_A_m",
        "rho0_kg_per_m3",
    ]
    for mode_curve in frequency_curves:
        header.extend(
            [
                "f_{}_Hz".format(mode_curve.mode),
                "omega_{}_rad_per_s".format(mode_curve.mode),
            ]
        )

    with out_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for i in range(len(curve.T0)):
            row = [
                float(curve.T0[i]),
                float(curve.T0[i] / mu),
                float(curve.lambda_R[i]),
                float(curve.lambda_Z[i]),
                float(curve.J1[i]),
                float(curve.J1[i] - 1.0),
                float(K_eff[i]),
                float(K_eff[i] / mu),
                float(radius_A),
                float(rho0),
            ]
            for mode_curve in frequency_curves:
                row.extend(
                    [
                        float(mode_curve.frequency_hz[i]),
                        float(mode_curve.omega[i]),
                    ]
                )
            writer.writerow(row)

    return out_path


def save_summary_text(prefix: str, summary_lines: Sequence[str]) -> Path:
    out_path = Path(prefix + "_summary.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(summary_lines) + "\n")
    return out_path


def make_frequency_figure(T0_over_mu: np.ndarray, frequency_curves: Sequence[FrequencyCurve]):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for mode_curve in frequency_curves:
        ax.plot(T0_over_mu, mode_curve.frequency_hz, label="mode {}".format(mode_curve.mode))

    ax.set_xlabel("T0 / mu")
    ax.set_ylabel("Frequence (Hz)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def make_stiffness_figure(T0_over_mu: np.ndarray, K_eff_over_mu: np.ndarray):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(T0_over_mu, K_eff_over_mu, color="black")
    ax.set_xlabel("T0 / mu")
    ax.set_ylabel("K_eff / mu")
    ax.grid(True)
    fig.tight_layout()
    return fig


def save_figures(prefix: str, fig_freq, fig_stiffness) -> Tuple[Path, Path]:
    freq_path = Path(prefix + "_freq_vs_T0_over_mu.png")
    stiffness_path = Path(prefix + "_Keff_vs_T0_over_mu.png")
    freq_path.parent.mkdir(parents=True, exist_ok=True)
    fig_freq.savefig(str(freq_path), dpi=180)
    fig_stiffness.savefig(str(stiffness_path), dpi=180)
    return freq_path, stiffness_path


def main():
    mu, kappa = pick_material_parameters()
    modes = validate_modes(MODES)

    T_max, dT0, dT_max = load_traction_preset(LOAD_PRESET, mu)

    prestrain_curve = solve_prestrain_continuation(mu, kappa, T_max, dT0, dT_max)
    K_eff = compute_effective_stiffness(prestrain_curve, mu)
    modal_frequencies = compute_frequency_curves(modes, RADIUS, RHO0, K_eff)

    print("mu =", mu, "kappa =", kappa)
    print("A(radius) =", RADIUS, "rho0 =", RHO0)
    print("load_preset =", LOAD_PRESET, "T0_max/mu =", T_max / mu)
    print("Formule rapport: Keff = T0/lambda_R + mu*J1^(-2/3)*lambda_Z^2/lambda_R^2")

    summary_lines = build_summary_lines(prestrain_curve, mu, K_eff, modal_frequencies)
    for line in summary_lines:
        print(line)

    if EXPORT_CSV is not None:
        csv_path = export_frequency_csv(
            path=EXPORT_CSV,
            curve=prestrain_curve,
            mu=mu,
            K_eff=K_eff,
            radius_A=RADIUS,
            rho0=RHO0,
            frequency_curves=modal_frequencies,
        )
        print("CSV export:", csv_path)

    T0_over_mu = prestrain_curve.T0 / mu
    fig_freq = make_frequency_figure(T0_over_mu, modal_frequencies)
    fig_stiffness = make_stiffness_figure(T0_over_mu, K_eff / mu)

    if SAVE_PREFIX is not None:
        figure_paths = save_figures(SAVE_PREFIX, fig_freq, fig_stiffness)
        summary_path = save_summary_text(SAVE_PREFIX, summary_lines)
        print("Figures exportees:", figure_paths[0], figure_paths[1])
        print("Resume exporte:", summary_path)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig_freq)
        plt.close(fig_stiffness)


if __name__ == "__main__":
    main()