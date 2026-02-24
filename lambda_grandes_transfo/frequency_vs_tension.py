from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from models import NeoHookeanReducedI1, drum_material_preset, mu_K_from_E_nu
from solvers import ContinuationSettings, NewtonSettings, continuation_in_T


# Zeros j_{n,m} of Bessel J_n used in the modal formula.
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
    """Continuation result for the preloaded state."""

    T0: np.ndarray
    lambda_R: np.ndarray
    lambda_Z: np.ndarray
    J1: np.ndarray


@dataclass
class FrequencyCurve:
    """Modal frequency data for one mode (n,m)."""

    mode: str
    alpha_nm: float
    omega: np.ndarray
    frequency_hz: np.ndarray


def load_traction_preset(name: str, mu: float) -> Tuple[float, float, float]:
    """Return (T0_max, dT0_init, dT0_max) from a preset expressed as fractions of mu."""
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Frequences d'un tambour circulaire f_{n,m}(T0) avec coefficient effectif "
            "K(T0) dependant de lambda_R(T0), lambda_Z(T0) (formule du rapport)."
        )
    )

    parser.add_argument("--mu", type=float, default=1.0, help="Module de cisaillement mu (Pa)")
    parser.add_argument("--Kbulk", type=float, default=100.0, help="Module volumique kappa (Pa)")
    parser.add_argument("--E", type=float, default=None, help="Module de Young (Pa)")
    parser.add_argument("--nu", type=float, default=None, help="Coefficient de Poisson")
    parser.add_argument("--preset", type=str, default=None, help="Preset materiau: mylar_92, steel_304")

    parser.add_argument(
        "--load-preset",
        type=str,
        default="drum_classic_tendu",
        help="Preset de traction: drum_classic_relax, drum_classic_tendu, drum_classic_tres_tendu",
    )
    parser.add_argument("--T-max", type=float, default=None, help="Traction nominale max T0 (Pa)")
    parser.add_argument("--dT0", type=float, default=None, help="Pas initial de continuation (Pa)")
    parser.add_argument("--dT-max", type=float, default=None, help="Pas max de continuation (Pa)")

    parser.add_argument("--radius", type=float, default=0.15, help="Rayon de reference A (m)")
    parser.add_argument(
        "--rho0",
        type=float,
        default=1400.0,
        help="Masse volumique referentielle rho0 (kg/m^3)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="01,11,21,02",
        help="Liste de modes separes par des virgules (ex: 01,11,21)",
    )

    parser.add_argument("--export-csv", type=str, default=None, help="CSV complet des courbes")
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="Prefixe des exports PNG/TXT (ex: results/coupled_mylar)",
    )
    parser.add_argument("--no-show", action="store_true", help="Ne pas afficher les figures")
    return parser.parse_args()


def parse_modes(text: str) -> List[str]:
    modes: List[str] = []
    for raw_mode in text.split(","):
        mode = raw_mode.strip()
        if not mode:
            continue
        if mode not in MODE_ROOTS:
            raise ValueError(
                "Mode inconnu '{}'. Modes disponibles: {}".format(mode, ", ".join(sorted(MODE_ROOTS)))
            )
        modes.append(mode)
    if not modes:
        raise ValueError("Aucun mode valide n'a ete fourni.")
    return modes


def pick_material_parameters(args) -> Tuple[float, float]:
    """Return (mu, kappa) from either direct values, (E, nu), or a preset."""
    if (args.E is None) ^ (args.nu is None):
        raise ValueError("Donner E et nu ensemble, ou aucun des deux.")
    if args.E is not None and args.nu is not None:
        return mu_K_from_E_nu(args.E, args.nu)
    if args.preset is not None:
        E, nu = drum_material_preset(args.preset)
        return mu_K_from_E_nu(E, nu)
    return float(args.mu), float(args.Kbulk)


def solve_prestrain_continuation(mu: float, kappa: float, T_max: float, dT0: float, dT_max: float) -> PrestrainCurve:
    """
    Solve the nonlinear prestrain problem:
      P_rR(lambda_R, lambda_Z) = T0
      P_zZ(lambda_R, lambda_Z) = 0
    using continuation in T0.
    """
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
    J1 = lambda_R ** 2 * lambda_Z
    return PrestrainCurve(T0=T0_values, lambda_R=lambda_R, lambda_Z=lambda_Z, J1=J1)


def compute_effective_stiffness(curve: PrestrainCurve, mu: float) -> np.ndarray:
    """
    Formula from the report:
      K(T0) = T0/lambda_R + mu * J1^(-2/3) * lambda_Z^2 / lambda_R^2
    """
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
    args = parse_args()
    mu, kappa = pick_material_parameters(args)
    modes = parse_modes(args.modes)

    T_preset, dT0_preset, dTmax_preset = load_traction_preset(args.load_preset, mu)
    T_max = float(args.T_max) if args.T_max is not None else float(T_preset)
    dT0 = float(args.dT0) if args.dT0 is not None else float(dT0_preset)
    dT_max = float(args.dT_max) if args.dT_max is not None else float(dTmax_preset)

    prestrain_curve = solve_prestrain_continuation(mu, kappa, T_max, dT0, dT_max)
    K_eff = compute_effective_stiffness(prestrain_curve, mu)
    modal_frequencies = compute_frequency_curves(modes, args.radius, args.rho0, K_eff)

    print("mu=", mu, "kappa=", kappa)
    print("A(radius)=", args.radius, "rho0=", args.rho0)
    print("load_preset=", args.load_preset, "T0_max/mu=", T_max / mu)
    print("Formule rapport: Keff = T0/lambda_R + mu*J1^(-2/3)*lambda_Z^2/lambda_R^2")

    summary_lines = build_summary_lines(prestrain_curve, mu, K_eff, modal_frequencies)
    for line in summary_lines:
        print(line)

    if args.export_csv:
        csv_path = export_frequency_csv(
            path=args.export_csv,
            curve=prestrain_curve,
            mu=mu,
            K_eff=K_eff,
            radius_A=float(args.radius),
            rho0=float(args.rho0),
            frequency_curves=modal_frequencies,
        )
        print("CSV export:", csv_path)

    T0_over_mu = prestrain_curve.T0 / mu
    fig_freq = make_frequency_figure(T0_over_mu, modal_frequencies)
    fig_stiffness = make_stiffness_figure(T0_over_mu, K_eff / mu)

    if args.save_prefix:
        figure_paths = save_figures(args.save_prefix, fig_freq, fig_stiffness)
        summary_path = save_summary_text(args.save_prefix, summary_lines)
        print("Figures exportees:", figure_paths[0], figure_paths[1])
        print("Resume exporte:", summary_path)

    if args.no_show:
        plt.close(fig_freq)
        plt.close(fig_stiffness)
    else:
        plt.show()


if __name__ == "__main__":
    main()
