from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from models import E_nu_from_mu_K, NeoHookeanReducedI1, drum_material_preset, mu_K_from_E_nu
from solvers import ContinuationSettings, NewtonSettings, continuation_in_T


@dataclass
class PreloadCurve:
    """Continuation results for the homogeneous preloaded state."""

    T0: np.ndarray
    lambda_R: np.ndarray
    lambda_Z: np.ndarray
    J1: np.ndarray


def load_traction_preset(name: str, mu: float) -> Tuple[float, float, float]:
    """Return (T0_max, dT_init, dT_max) for a preset, expressed as fractions of mu."""
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
        description="Continuation en traction nominale T0 pour un tambour axisymetrique hyperelastique."
    )
    parser.add_argument("--mu", type=float, default=1.0, help="Module de cisaillement mu (Pa)")
    parser.add_argument("--K", type=float, default=100.0, help="Module volumique kappa (Pa)")
    parser.add_argument("--E", type=float, default=None, help="Module de Young (Pa)")
    parser.add_argument("--nu", type=float, default=None, help="Coefficient de Poisson")
    parser.add_argument("--preset", type=str, default=None, help="Preset materiau: mylar_92, steel_304")

    parser.add_argument(
        "--load-preset",
        type=str,
        default="drum_classic_tendu",
        help="Preset de traction: drum_classic_relax, drum_classic_tendu, drum_classic_tres_tendu",
    )
    parser.add_argument("--T-max", type=float, default=None, help="Traction nominale maximale T0 (Pa)")
    parser.add_argument("--dT0", type=float, default=None, help="Pas initial de continuation (Pa)")
    parser.add_argument("--dT-max", type=float, default=None, help="Pas maximal de continuation (Pa)")

    parser.add_argument("--export-csv", type=str, default=None, help="CSV de la courbe de continuation")
    parser.add_argument(
        "--save-prefix",
        type=str,
        default=None,
        help="Prefixe pour les exports PNG/TXT (ex: results/demo)",
    )
    parser.add_argument("--no-show", action="store_true", help="Ne pas afficher les figures")
    return parser.parse_args()


def pick_material_parameters(args) -> Tuple[float, float]:
    """Resolve (mu, kappa) from direct values, (E,nu), or a material preset."""
    if (args.E is None) ^ (args.nu is None):
        raise ValueError("Donner E et nu ensemble, ou aucun des deux.")
    if args.E is not None and args.nu is not None:
        return mu_K_from_E_nu(args.E, args.nu)
    if args.preset is not None:
        E, nu = drum_material_preset(args.preset)
        return mu_K_from_E_nu(E, nu)
    return float(args.mu), float(args.K)


def solve_preload_curve(mu: float, kappa: float, T_max: float, dT_init: float, dT_max: float) -> PreloadCurve:
    """
    Solve the homogeneous preloaded state through continuation in T0.

    Unknowns are parameterized as log-stretches:
      x = [ln(lambda_R), ln(lambda_Z)]
    """
    law = NeoHookeanReducedI1(mu=mu, K=kappa)

    def make_residual(T0):
        def residual(log_lambda):
            lambda_R = float(np.exp(log_lambda[0]))
            lambda_Z = float(np.exp(log_lambda[1]))
            return np.array(
                [
                    law.P_rr(lambda_R, lambda_Z) - T0,
                    law.P_zz(lambda_R, lambda_Z),
                ],
                dtype=float,
            )

        return residual

    solution = continuation_in_T(
        make_residual=make_residual,
        stN=NewtonSettings(),
        stC=ContinuationSettings(T_max=T_max, dT0=dT_init, dT_max=dT_max),
        x_init=np.array([0.0, 0.0]),
    )

    T0_values = np.array([item[0] for item in solution], dtype=float)
    log_lambdas = np.array([item[1] for item in solution], dtype=float)
    lambda_R = np.exp(log_lambdas[:, 0])
    lambda_Z = np.exp(log_lambdas[:, 1])
    J1 = lambda_R ** 2 * lambda_Z

    return PreloadCurve(T0=T0_values, lambda_R=lambda_R, lambda_Z=lambda_Z, J1=J1)


def export_curve_csv(path: str, curve: PreloadCurve, mu: float) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "T0_Pa",
                "T0_over_mu",
                "lambda_R",
                "lambda_Z",
                "eps_R=lambda_R-1",
                "eps_Z=lambda_Z-1",
                "J1",
                "J1_minus_1",
            ]
        )
        for i in range(len(curve.T0)):
            writer.writerow(
                [
                    float(curve.T0[i]),
                    float(curve.T0[i] / mu),
                    float(curve.lambda_R[i]),
                    float(curve.lambda_Z[i]),
                    float(curve.lambda_R[i] - 1.0),
                    float(curve.lambda_Z[i] - 1.0),
                    float(curve.J1[i]),
                    float(curve.J1[i] - 1.0),
                ]
            )

    return out_path


def build_summary_lines(curve: PreloadCurve, mu: float) -> List[str]:
    idx_mid = len(curve.T0) // 2
    indices = [0, idx_mid, len(curve.T0) - 1]
    labels = ["debut", "milieu", "fin"]
    lines: List[str] = []

    for label, i in zip(labels, indices):
        lines.append(
            (
                "{}: T0={:.6e} Pa, T0/mu={:.6e}, lambda_R={:.8f}, "
                "lambda_Z={:.8f}, J1-1={:.3e}"
            ).format(
                label,
                curve.T0[i],
                curve.T0[i] / mu,
                curve.lambda_R[i],
                curve.lambda_Z[i],
                curve.J1[i] - 1.0,
            )
        )

    if len(curve.T0) >= 2:
        d_lambda_R_dT = (curve.lambda_R[1] - curve.lambda_R[0]) / (curve.T0[1] - curve.T0[0])
        d_lambda_Z_dT = (curve.lambda_Z[1] - curve.lambda_Z[0]) / (curve.T0[1] - curve.T0[0])
        lines.append(
            "pente initiale approx: d(lambda_R)/dT0={:.6e} Pa^-1, d(lambda_Z)/dT0={:.6e} Pa^-1".format(
                d_lambda_R_dT, d_lambda_Z_dT
            )
        )

    lines.append("max |J1-1| = {:.6e}".format(float(np.max(np.abs(curve.J1 - 1.0)))))
    return lines


def save_summary_text(prefix: str, lines: Sequence[str]) -> Path:
    out_path = Path(prefix + "_summary.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def make_stretch_figure(curve: PreloadCurve, mu: float):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    T0_over_mu = curve.T0 / mu
    ax.plot(T0_over_mu, curve.lambda_R, label="lambda_R")
    ax.plot(T0_over_mu, curve.lambda_Z, label="lambda_Z")
    ax.set_xlabel("T0 / mu")
    ax.set_ylabel("Stretches")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def make_volume_figure(curve: PreloadCurve, mu: float):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(curve.T0 / mu, curve.J1 - 1.0, color="black", label="J1 - 1")
    ax.set_xlabel("T0 / mu")
    ax.set_ylabel("J1 - 1")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def save_figures(prefix: str, fig_stretches, fig_volume) -> Tuple[Path, Path]:
    stretches_path = Path(prefix + "_lr_lz.png")
    volume_path = Path(prefix + "_J_minus_1.png")
    stretches_path.parent.mkdir(parents=True, exist_ok=True)
    fig_stretches.savefig(str(stretches_path), dpi=180)
    fig_volume.savefig(str(volume_path), dpi=180)
    return stretches_path, volume_path


def main():
    args = parse_args()
    mu, kappa = pick_material_parameters(args)

    T_preset, dT_preset, dTmax_preset = load_traction_preset(args.load_preset, mu)
    T_max = float(args.T_max) if args.T_max is not None else float(T_preset)
    dT_init = float(args.dT0) if args.dT0 is not None else float(dT_preset)
    dT_max = float(args.dT_max) if args.dT_max is not None else float(dTmax_preset)

    curve = solve_preload_curve(mu, kappa, T_max, dT_init, dT_max)

    E_eff, nu_eff = E_nu_from_mu_K(mu, kappa)
    print("mu=", mu, "kappa=", kappa)
    print("E=", E_eff, "nu=", nu_eff)
    print("load_preset=", args.load_preset)
    print(
        "T0_max(Pa)=",
        T_max,
        "dT0(Pa)=",
        dT_init,
        "dT_max(Pa)=",
        dT_max,
        "T0_max/mu=",
        T_max / mu,
    )
    if T_max / mu < 0.1:
        print("Warning: T0_max/mu < 0.1 => reponse attendue quasi-lineaire.")
    print("Last:", curve.T0[-1], curve.lambda_R[-1], curve.lambda_Z[-1], curve.J1[-1])

    summary_lines = build_summary_lines(curve, mu)
    for line in summary_lines:
        print(line)

    if args.export_csv:
        csv_path = export_curve_csv(args.export_csv, curve, mu)
        print("CSV export:", csv_path)

    fig_stretches = make_stretch_figure(curve, mu)
    fig_volume = make_volume_figure(curve, mu)

    if args.save_prefix:
        stretch_path, volume_path = save_figures(args.save_prefix, fig_stretches, fig_volume)
        summary_path = save_summary_text(args.save_prefix, summary_lines)
        print("Figures exportees:", stretch_path, volume_path)
        print("Resume exporte:", summary_path)

    if args.no_show:
        plt.close(fig_stretches)
        plt.close(fig_volume)
    else:
        plt.show()


if __name__ == "__main__":
    main()
