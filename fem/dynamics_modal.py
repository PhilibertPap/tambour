from __future__ import annotations

from pathlib import Path
import wave

import numpy as np


# ==========================================================
# PARAMETRES (modifier ici puis executer)
# ==========================================================

# Donnees modales exportees par fem/main.py
MODAL_DATA_FILE = "fem/results/cercle/cercle_modal_data.npz"

# Selection de modes (utile si le 1er mode parait parasite / bizarre)
SKIP_FIRST_MODE = True
MIN_FREQ_HZ = 30.0
MAX_FREQ_HZ = 2.0e4
MAX_MODES_USED = 80   # plus de modes => son plus riche (si disponibles)

# Frappe (position dans le plan)
X_HIT = 0.04   # m (ni trop centre, ni trop bord)
Y_HIT = 0.0   # m
HIT_RADIUS = 0.002  # m (plus local => excite plus de modes)

# Force temporelle (impulsion gaussienne)
FORCE_AMPLITUDE = 1.0      # amplitude arbitraire (affecte amplitude, pas frequences)
T_HIT = 0.02               # s
TAU_HIT = 0.00035          # s (frappe plus courte => plus de hautes frequences)

# Amortissement modal
DAMPING_RATIO = 0.005       # amortissement modal constant (compromis)

# Point "micro" pour le signal audio
X_MIC = 0.02
Y_MIC = 0.01
MIC_RADIUS = 0.02            # m (micro spatialise => evite de tomber sur un noeud modal)
USE_VELOCITY_FOR_AUDIO = False
USE_ACCELERATION_FOR_AUDIO = True  # si True, prioritaire
AUDIO_FADEIN_AFTER_HIT = 0.0     # s (attenue le clic, laisse le "ring")
AUDIO_MUTE_BEFORE = 0.021        # s (coupe le "poc" direct si besoin)
AUDIO_NORMALIZE_AFTER = 0.023    # s (normalise sur la queue vibratoire)

# Temps
FS_AUDIO = 44100           # Hz
T_END = 2.0               # s

# Exports
WRITE_SIGNAL_CSV = True
WRITE_WAV = True
WRITE_VTK_ANIMATION = True    # mettre False si tu veux juste le son
N_VTK_SNAPSHOTS = 80
VTK_ANIMATION_SCALE = 1.0e-3   # m (amplitude visuelle des snapshots)
VTK_T_START = None             # None => commence a T_HIT (pas de "temps morts" avant l'impact)
VTK_T_END = 0.06               # s (fenetre exportee)
VTK_DENSE_AT_END = True        # snapshots plus serres plus tard (ring), pas au tout debut


def main():
    modal_path = Path(MODAL_DATA_FILE)
    if not modal_path.exists():
        # Petit confort "TD": on cherche automatiquement un fichier *_modal_data.npz
        candidates = sorted(Path("fem/results").glob("*/*_modal_data.npz"))
        if len(candidates) == 1:
            modal_path = candidates[0]
            print("MODAL_DATA_FILE introuvable, fichier detecte automatiquement :", modal_path)
        else:
            print("Fichier modal introuvable :", MODAL_DATA_FILE)
            if candidates:
                print("Fichiers *_modal_data.npz disponibles :")
                for p in candidates:
                    print("  -", p)
                print("Modifie MODAL_DATA_FILE en haut du script.")
            else:
                print("Aucun fichier *_modal_data.npz trouve.")
                print("Relance d'abord : python fem/main.py")
            return

    data = np.load(modal_path, allow_pickle=True)

    omegas_all = np.asarray(data["omegas"], dtype=float)              # (Nm,)
    freqs_hz_all = np.asarray(data["freqs_hz"], dtype=float)          # (Nm,)
    Phi_all = np.asarray(data["mode_vectors_W"], dtype=float)         # (Nm, Nw), normalises masse
    Xw = np.asarray(data["dof_coords_W"], dtype=float)                # (Nw, dim)
    mesh_file = str(np.asarray(data["mesh_file"]).ravel()[0])
    fe_degree = int(np.asarray(data["fe_degree"]).ravel()[0])
    case_name = str(np.asarray(data["case_name"]).ravel()[0])

    Nm_all, Nw = Phi_all.shape
    if Xw.shape[0] != Nw:
        raise RuntimeError("Incoherence NPZ: nombre de coordonnees de ddl != taille des modes.")

    # Selection simple des modes (style TD)
    keep = np.ones(Nm_all, dtype=bool)
    if SKIP_FIRST_MODE and Nm_all > 0:
        keep[0] = False
    keep &= (freqs_hz_all >= float(MIN_FREQ_HZ))
    keep &= (freqs_hz_all <= float(MAX_FREQ_HZ))
    ids = np.nonzero(keep)[0]
    if MAX_MODES_USED is not None:
        ids = ids[: int(MAX_MODES_USED)]
    if len(ids) == 0:
        raise RuntimeError("Aucun mode retenu (regler SKIP_FIRST_MODE / MIN_FREQ_HZ / MAX_MODES_USED).")

    omegas = omegas_all[ids]
    freqs_hz = freqs_hz_all[ids]
    Phi = Phi_all[ids, :]
    Nm = len(ids)

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

    # Point micro (ddl le plus proche + moyenne spatiale pour robustesse audio)
    r2_mic = (x - X_MIC) ** 2 + (y - Y_MIC) ** 2
    i_mic = int(np.argmin(r2_mic))
    i_hit = int(np.argmin(r2_hit))

    if MIC_RADIUS > 0.0:
        mic_weights = np.exp(-0.5 * r2_mic / (MIC_RADIUS ** 2))
        wsum_mic = float(np.sum(np.abs(mic_weights)))
        if wsum_mic > 0.0:
            mic_weights /= wsum_mic
        else:
            mic_weights[:] = 0.0
            mic_weights[i_mic] = 1.0
    else:
        mic_weights = np.zeros_like(x)
        mic_weights[i_mic] = 1.0

    # Force modale f_n(t) = g_n * A(t) (modes normalises en masse)
    g_modal = Phi @ spatial_hit

    # Valeur modale au "micro" (moyenne spatiale)
    phi_mic = Phi @ mic_weights

    print("=== Dynamics modal (TD) ===")
    print("MODAL_DATA_FILE =", modal_path)
    print("mesh_file =", mesh_file)
    print("Nmodes disponibles =", Nm_all)
    print("Nmodes utilises =", Nm)
    print("f1.. =", ", ".join(f"{f:.1f}" for f in freqs_hz[: min(6, len(freqs_hz))]), "Hz")
    print("Point impact cible = ({:.4f}, {:.4f}) m".format(X_HIT, Y_HIT))
    print("Point impact ddl   = ({:.4f}, {:.4f}) m".format(float(x[i_hit]), float(y[i_hit])))
    print("Point micro cible  = ({:.4f}, {:.4f}) m".format(X_MIC, Y_MIC))
    print("Point micro ddl    = ({:.4f}, {:.4f}) m".format(float(x[i_mic]), float(y[i_mic])))
    print("MIC_RADIUS =", MIC_RADIUS, "m")
    print("Couplage impact max|g_modal| =", float(np.max(np.abs(g_modal))))
    print("Couplage micro  max|phi_mic| =", float(np.max(np.abs(phi_mic))))
    modal_gain = np.abs(g_modal * phi_mic)
    if modal_gain.size:
        idx_sort = np.argsort(-modal_gain)
        gmax = float(np.max(modal_gain)) if float(np.max(modal_gain)) > 0 else 1.0
        print("Modes les plus excites (freq Hz / gain rel.):")
        for j in idx_sort[: min(8, len(idx_sort))]:
            print(f"  f={freqs_hz[j]:8.1f} Hz, gain={modal_gain[j]/gmax:.3e}")

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
    a_mic = np.zeros(Nt, dtype=float)

    # Snapshots VTK (optionnel)
    snapshot_ids = None
    q_snapshots = None
    if WRITE_VTK_ANIMATION:
        t_snap_start = float(T_HIT) if VTK_T_START is None else float(VTK_T_START)
        t_snap_start = max(0.0, min(t_snap_start, float(T_END)))
        t_snap_end = max(t_snap_start, min(float(VTK_T_END), float(T_END)))
        if VTK_DENSE_AT_END:
            # plus grossier juste apres l'impact, plus fin ensuite
            s = np.linspace(0.0, 1.0, int(N_VTK_SNAPSHOTS))
            t_snap = t_snap_start + np.sqrt(s) * (t_snap_end - t_snap_start)
        else:
            t_snap = np.linspace(t_snap_start, t_snap_end, int(N_VTK_SNAPSHOTS))
        snapshot_ids = np.unique(np.clip(np.round(t_snap / dt).astype(int), 0, Nt - 1))
        q_snapshots = np.zeros((len(snapshot_ids), Nm), dtype=float)
        k_snap = 0

    def force_time(tt):
        return FORCE_AMPLITUDE * np.exp(-0.5 * ((tt - T_HIT) / TAU_HIT) ** 2)

    for k in range(Nt):
        tk = t[k]

        # Stockage signal avant update (etat courant)
        f_now = g_modal * force_time(tk)
        qdd_now = f_now - damping * qd - stiffness * q
        w_mic[k] = float(np.dot(q, phi_mic))
        v_mic[k] = float(np.dot(qd, phi_mic))
        a_mic[k] = float(np.dot(qdd_now, phi_mic))

        if WRITE_VTK_ANIMATION and snapshot_ids is not None and q_snapshots is not None:
            if k_snap < len(snapshot_ids) and k == snapshot_ids[k_snap]:
                q_snapshots[k_snap, :] = q
                k_snap += 1

        # RK4 explicite (style TD)
        f1 = g_modal * force_time(tk)
        k1_q = qd
        k1_v = f1 - damping * qd - stiffness * q

        q2 = q + 0.5 * dt * k1_q
        v2 = qd + 0.5 * dt * k1_v
        f2 = g_modal * force_time(tk + 0.5 * dt)
        k2_q = v2
        k2_v = f2 - damping * v2 - stiffness * q2

        q3 = q + 0.5 * dt * k2_q
        v3 = qd + 0.5 * dt * k2_v
        f3 = g_modal * force_time(tk + 0.5 * dt)
        k3_q = v3
        k3_v = f3 - damping * v3 - stiffness * q3

        q4 = q + dt * k3_q
        v4 = qd + dt * k3_v
        f4 = g_modal * force_time(tk + dt)
        k4_q = v4
        k4_v = f4 - damping * v4 - stiffness * q4

        q = q + (dt / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
        qd = qd + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    # ------------------------------------------------------
    # Signal audio (point micro)
    # ------------------------------------------------------
    if USE_ACCELERATION_FOR_AUDIO:
        signal = a_mic.copy()
    elif USE_VELOCITY_FOR_AUDIO:
        signal = v_mic.copy()
    else:
        signal = w_mic.copy()

    if AUDIO_FADEIN_AFTER_HIT > 0.0:
        gate = np.ones_like(signal)
        mask = t >= float(T_HIT)
        gate[mask] = 1.0 - np.exp(-(t[mask] - float(T_HIT)) / float(AUDIO_FADEIN_AFTER_HIT))
        signal *= gate

    if AUDIO_MUTE_BEFORE > 0.0:
        signal = signal.copy()
        signal[t < float(AUDIO_MUTE_BEFORE)] = 0.0

    signal = signal - np.mean(signal)
    if AUDIO_NORMALIZE_AFTER is not None:
        mask_norm = t >= float(AUDIO_NORMALIZE_AFTER)
        signal_ref = signal[mask_norm] if np.any(mask_norm) else signal
    else:
        signal_ref = signal

    peak = float(np.max(np.abs(signal_ref))) if signal_ref.size else 0.0
    rms = float(np.sqrt(np.mean(signal ** 2))) if signal.size else 0.0
    print("Signal brut: peak = {:.6e}, rms = {:.6e}".format(peak, rms))
    if peak > 0:
        signal_norm = 0.95 * signal / peak
    else:
        signal_norm = signal.copy()
        print("warning: signal nul (ou quasi nul) -> WAV silencieux")

    case_dir = modal_path.parent
    stem = modal_path.stem.replace("_modal_data", "")
    csv_signal_path = case_dir / f"{stem}_hit_signal.csv"
    wav_path = case_dir / f"{stem}_hit.wav"
    vtk_anim_path = case_dir / f"{stem}_hit_response.pvd"

    if WRITE_SIGNAL_CSV:
        with csv_signal_path.open("w") as f:
            f.write("t_s,w_mic,v_mic,a_mic,signal_audio\n")
            for ti, wi, vi, ai, si in zip(t, w_mic, v_mic, a_mic, signal_norm):
                f.write(f"{ti:.9e},{wi:.9e},{vi:.9e},{ai:.9e},{si:.9e}\n")
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
