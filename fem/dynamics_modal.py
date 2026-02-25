from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


# ==========================================================
# PARAMETRES (modifier ici puis executer)
# ==========================================================

# Donnees modales exportees par fem/main.py
MODAL_DATA_FILE = "fem/results/cercle/cercle_modal_data.npz"

# Frappe (position dans le plan)
X_HIT = 0.03   # m
Y_HIT = 0.02   # m
HIT_RADIUS = 0.01  # m (gaussienne spatiale)

# Force temporelle (impulsion gaussienne)
FORCE_AMPLITUDE = 1.0      # amplitude arbitraire (affecte l'amplitude, pas les frequences)
T_HIT = 0.02               # s
TAU_HIT = 0.0015           # s (duree de l'impulsion)

# Amortissement modal
DAMPING_RATIO = 0.01       # amortissement modal constant (1%)

# Point "micro" pour le signal audio
X_MIC = 0.04
Y_MIC = 0.00
USE_VELOCITY_FOR_AUDIO = True

# Temps
FS_AUDIO = 44100           # Hz
T_END = 2.0               # s

# Exports
WRITE_SIGNAL_CSV = True
WRITE_WAV = True
WRITE_VTK_ANIMATION = False    # mettre True si tu veux des snapshots ParaView
N_VTK_SNAPSHOTS = 80
VTK_ANIMATION_SCALE = 1.0e-3   # m (amplitude visuelle des snapshots)


def main():
    data = np.load(MODAL_DATA_FILE, allow_pickle=True)

    omegas = np.asarray(data["omegas"], dtype=float)                  # (Nm,)
    freqs_hz = np.asarray(data["freqs_hz"], dtype=float)              # (Nm,)
    Phi = np.asarray(data["mode_vectors_W"], dtype=float)             # (Nm, Nw), normalises masse
    Xw = np.asarray(data["dof_coords_W"], dtype=float)                # (Nw, dim)
    mesh_file = str(np.asarray(data["mesh_file"]).ravel()[0])
    fe_degree = int(np.asarray(data["fe_degree"]).ravel()[0])
    case_name = str(np.asarray(data["case_name"]).ravel()[0])

    Nm, Nw = Phi.shape
    if Xw.shape[0] != Nw:
        raise RuntimeError("Incoherence NPZ: nombre de coordonnees de ddl != taille des modes.")

    # ------------------------------------------------------
    # Spatialisation de la frappe (gaussienne sur les ddl scalaires)
    # ------------------------------------------------------
    x = Xw[:, 0]
    y = Xw[:, 1]
    r2_hit = (x - X_HIT) ** 2 + (y - Y_HIT) ** 2
    if HIT_RADIUS > 0:
        spatial_hit = np.exp(-0.5 * r2_hit / (HIT_RADIUS ** 2))
    else:
        spatial_hit = np.zeros_like(x)
        spatial_hit[int(np.argmin(r2_hit))] = 1.0

    # Normalisation simple du profil spatial (integrale discrete = 1)
    ssum = float(np.sum(np.abs(spatial_hit)))
    if ssum <= 0:
        raise RuntimeError("Profil de frappe nul.")
    spatial_hit /= ssum

    # Point micro (ddl le plus proche)
    r2_mic = (x - X_MIC) ** 2 + (y - Y_MIC) ** 2
    i_mic = int(np.argmin(r2_mic))
    i_hit = int(np.argmin(r2_hit))

    # Force modale f_n(t) = g_n * A(t) (modes normalises en masse)
    g_modal = Phi @ spatial_hit

    # Valeur modale au point micro pour la reconstruction du signal
    phi_mic = Phi[:, i_mic]

    print("=== Dynamics modal (TD) ===")
    print("MODAL_DATA_FILE =", MODAL_DATA_FILE)
    print("mesh_file =", mesh_file)
    print("Nmodes utilises =", Nm)
    print("f1.. =", ", ".join(f"{f:.1f}" for f in freqs_hz[: min(6, len(freqs_hz))]), "Hz")
    print("Point impact cible = ({:.4f}, {:.4f}) m".format(X_HIT, Y_HIT))
    print("Point impact ddl   = ({:.4f}, {:.4f}) m".format(float(x[i_hit]), float(y[i_hit])))
    print("Point micro cible  = ({:.4f}, {:.4f}) m".format(X_MIC, Y_MIC))
    print("Point micro ddl    = ({:.4f}, {:.4f}) m".format(float(x[i_mic]), float(y[i_mic])))

    # ------------------------------------------------------
    # Integration temporelle modale (RK4 vectorise, simple)
    # q'' + 2 zeta w q' + w^2 q = g * A(t)
    # ------------------------------------------------------
    dt = 1.0 / float(FS_AUDIO)
    t = np.arange(0.0, float(T_END), dt)
    Nt = len(t)

    q = np.zeros(Nm, dtype=float)
    qd = np.zeros(Nm, dtype=float)

    zeta = float(DAMPING_RATIO)
    damping = 2.0 * zeta * omegas
    stiffness = omegas ** 2

    w_mic = np.zeros(Nt, dtype=float)
    v_mic = np.zeros(Nt, dtype=float)

    # Snapshots VTK (optionnel)
    snapshot_ids = None
    q_snapshots = None
    if WRITE_VTK_ANIMATION:
        snapshot_ids = np.unique(np.linspace(0, Nt - 1, int(N_VTK_SNAPSHOTS), dtype=int))
        q_snapshots = np.zeros((len(snapshot_ids), Nm), dtype=float)
        k_snap = 0

    def force_time(tt):
        return FORCE_AMPLITUDE * np.exp(-0.5 * ((tt - T_HIT) / TAU_HIT) ** 2)

    def rhs(tt, q_, qd_):
        f_t = g_modal * force_time(tt)
        qdd_ = f_t - damping * qd_ - stiffness * q_
        return qd_, qdd_

    for k in range(Nt):
        tk = t[k]

        # Stockage signal avant update (etat courant)
        w_mic[k] = float(np.dot(q, phi_mic))
        v_mic[k] = float(np.dot(qd, phi_mic))

        if WRITE_VTK_ANIMATION and snapshot_ids is not None and q_snapshots is not None:
            if k_snap < len(snapshot_ids) and k == snapshot_ids[k_snap]:
                q_snapshots[k_snap, :] = q
                k_snap += 1

        # RK4 sur l'etat modal [q, qd]
        k1_q, k1_v = rhs(tk, q, qd)
        k2_q, k2_v = rhs(tk + 0.5 * dt, q + 0.5 * dt * k1_q, qd + 0.5 * dt * k1_v)
        k3_q, k3_v = rhs(tk + 0.5 * dt, q + 0.5 * dt * k2_q, qd + 0.5 * dt * k2_v)
        k4_q, k4_v = rhs(tk + dt, q + dt * k3_q, qd + dt * k3_v)

        q = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        qd = qd + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    # ------------------------------------------------------
    # Signal audio (point micro)
    # ------------------------------------------------------
    signal = v_mic if USE_VELOCITY_FOR_AUDIO else w_mic
    signal = signal - np.mean(signal)
    peak = float(np.max(np.abs(signal)))
    if peak > 0:
        signal_norm = 0.95 * signal / peak
    else:
        signal_norm = signal.copy()

    modal_path = Path(MODAL_DATA_FILE)
    case_dir = modal_path.parent
    stem = modal_path.stem.replace("_modal_data", "")
    csv_signal_path = case_dir / f"{stem}_hit_signal.csv"
    wav_path = case_dir / f"{stem}_hit.wav"
    vtk_anim_path = case_dir / f"{stem}_hit_response.pvd"

    if WRITE_SIGNAL_CSV:
        with csv_signal_path.open("w") as f:
            f.write("t_s,w_mic,v_mic,signal_audio\n")
            for ti, wi, vi, si in zip(t, w_mic, v_mic, signal_norm):
                f.write(f"{ti:.9e},{wi:.9e},{vi:.9e},{si:.9e}\n")
        print("CSV signal exporte:", csv_signal_path)

    if WRITE_WAV:
        samples = np.asarray(np.clip(signal_norm, -1.0, 1.0) * 32767.0, dtype=np.int16)
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16
            wf.setframerate(int(FS_AUDIO))
            wf.writeframes(samples.tobytes())
        print("WAV exporte:", wav_path)

    # ------------------------------------------------------
    # Snapshots VTK (optionnel) pour ParaView
    # ------------------------------------------------------
    if WRITE_VTK_ANIMATION and snapshot_ids is not None and q_snapshots is not None:
        from mpi4py import MPI  # type: ignore
        from dolfinx import fem, io  # type: ignore
        import ufl  # type: ignore

        # Relecture du maillage (compat versions dolfinx)
        try:
            from dolfinx.io import gmshio  # type: ignore
            domain, _, _ = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, rank=0, gdim=3)
        except Exception:
            out = io.gmsh.read_from_msh(mesh_file, MPI.COMM_WORLD, rank=0, gdim=3)
            if not isinstance(out, (tuple, list)) or len(out) < 3:
                raise RuntimeError("API dolfinx.io.gmsh.read_from_msh inattendue.")
            domain = out[0]

        try:
            V = fem.functionspace(domain, ("Lagrange", fe_degree, (3,)))
        except Exception:
            V = fem.FunctionSpace(domain, ufl.VectorElement("Lagrange", domain.ufl_cell(), fe_degree))

        W, map_W_to_Vz_now = V.sub(2).collapse()
        if len(map_W_to_Vz_now) != Nw:
            raise RuntimeError("Le nombre de ddl de W a change; impossible de reconstruire les snapshots VTK.")

        u_dyn = fem.Function(V, name="u_dyn")
        with io.VTKFile(domain.comm, str(vtk_anim_path), "w") as vtk:
            vtk.write_mesh(domain)
            for i_snap, idx in enumerate(snapshot_ids):
                w_field = q_snapshots[i_snap, :] @ Phi  # (Nw,)
                # amplitude visuelle
                vmax = float(np.max(np.abs(w_field)))
                if vmax > 0:
                    w_field = (VTK_ANIMATION_SCALE / vmax) * w_field

                u_dyn.x.array[:] = 0.0
                u_dyn.x.array[np.asarray(map_W_to_Vz_now, dtype=np.int64)] = w_field
                u_dyn.x.scatter_forward()
                vtk.write_function(u_dyn, float(t[idx]))
        print("VTK animation exporte:", vtk_anim_path)


if __name__ == "__main__":
    main()
