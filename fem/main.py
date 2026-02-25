from __future__ import annotations

from pathlib import Path

import numpy as np


# ==========================================================
# PARAMETRES
# ==========================================================

# Maillage / tags (les scripts dans fem/mesh utilisent ces tags)
MESH_FILE = "fem/mesh/cercle.msh"
TAG_LATERAL = 10   # Gamma_lat
TAG_FIX_FACE = 12  # utilise seulement si GAUGE_MODE == "face"

# Jauge pour supprimer les mouvements de corps rigide
# - "minimal" : 3 points / 6 ddl (recommande, perturbe peu la physique)
# - "face"    : bloque une face complete (simple mais plus intrusif)
GAUGE_MODE = "minimal"  # "minimal" ou "face"

# Materiau (choisir preset OU imposer E, nu, rho0)
USE_PRESET = True
PRESET = "mylar_92"  # "mylar_92" ou "steel_304"
E = 3.5e9            # Pa (ignore si USE_PRESET=True)
NU = 0.38            # -
RHO0 = 1400.0        # kg/m^3

# Chargement
T0_TARGET = 1.0e6    # Pa (traction nominale sur Gamma_lat)
N_LOAD_STEPS = 50    # continuation simple (plus grand = plus robuste)

# Discretisation / solveurs
FE_DEGREE = 2
NEWTON_ATOL = 1e-8
NEWTON_RTOL = 5e-4   # tolerance relative par pas de charge (plus robuste)
NEWTON_MAX_IT = 40
N_MODES = 10
EIG_TARGET = 0.0

# Exports (dans fem/results)
WRITE_CSV = True
WRITE_VTK = True
RESULTS_BASENAME = None  # ex: "cercle_T1e6"; None => derive du nom de maillage


# ==========================================================
# Petites fonctions utilitaires (minimum)
# ==========================================================


def material_preset(name: str):
    key = name.strip().lower()
    if key == "mylar_92":
        return 3.5e9, 0.38, 1400.0
    if key == "steel_304":
        return 193e9, 0.29, 7900.0
    raise ValueError("Preset inconnu. Choix: mylar_92, steel_304.")


def mu_kappa_from_E_nu(E: float, nu: float):
    return E / (2.0 * (1.0 + nu)), E / (3.0 * (1.0 - 2.0 * nu))


# ==========================================================
# Programme principal
# ==========================================================


def main():
    from mpi4py import MPI  
    from petsc4py import PETSc  
    from slepc4py import SLEPc  
    from dolfinx import fem, io  
    from dolfinx.fem import petsc as fem_petsc 
    import ufl 

    # ------------------------------------------------------
    # Parametres materiau
    # ------------------------------------------------------
    if USE_PRESET:
        E_val, nu_val, rho0_val = material_preset(PRESET)
    else:
        E_val, nu_val, rho0_val = E, NU, RHO0
    mu, kappa = mu_kappa_from_E_nu(float(E_val), float(nu_val))

    # ------------------------------------------------------
    # Lecture maillage Gmsh (.msh) 
    # ------------------------------------------------------
    try:
        from dolfinx.io import gmshio 
        domain, cell_tags, facet_tags = gmshio.read_from_msh(MESH_FILE, MPI.COMM_WORLD, rank=0, gdim=3)
    except Exception:
        if hasattr(io, "gmsh") and hasattr(io.gmsh, "read_from_msh"):
            out = io.gmsh.read_from_msh(MESH_FILE, MPI.COMM_WORLD, rank=0, gdim=3)
            if not isinstance(out, (tuple, list)) or len(out) < 3:
                raise RuntimeError("API dolfinx.io.gmsh.read_from_msh inattendue.")
            domain, cell_tags, facet_tags = out[0], out[1], out[2]
        else:
            raise RuntimeError("Impossible de lire le maillage .msh (gmshio/read_from_msh introuvable).")

    # ------------------------------------------------------
    # Espace EF vectoriel
    # ------------------------------------------------------
    try:
        V = fem.functionspace(domain, ("Lagrange", FE_DEGREE, (3,)))
    except Exception:
        V = fem.FunctionSpace(domain, ufl.VectorElement("Lagrange", domain.ufl_cell(), FE_DEGREE))

    u1 = fem.Function(V, name="u_prestress")
    v = ufl.TestFunction(V)
    du = ufl.TrialFunction(V)

    # ------------------------------------------------------
    # Loi hyperelastique (reduced-I1 neo-hookean + volumique)
    # ------------------------------------------------------
    I = ufl.Identity(3)
    F1 = I + ufl.grad(u1)
    F1_var = ufl.variable(F1)
    C1 = F1_var.T * F1_var
    J1 = ufl.det(F1_var)
    I1 = ufl.tr(C1)
    I1_bar = J1 ** (-2.0 / 3.0) * I1
    psi = 0.5 * mu * (I1_bar - 3.0) + 0.25 * kappa * (J1 * J1 - 1.0 - 2.0 * ufl.ln(J1))
    P1 = ufl.diff(psi, F1_var)

    # ------------------------------------------------------
    # Prechargement non lineaire
    #    \int P(F1):grad(v) dx = \int_{Gamma_lat} T0 N.v ds
    # ------------------------------------------------------
    N = ufl.FacetNormal(domain)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    T0 = fem.Constant(domain, PETSc.ScalarType(0.0))
    traction = T0 * N

    R_form = ufl.inner(P1, ufl.grad(v)) * ufl.dx - ufl.inner(traction, v) * ds(TAG_LATERAL)
    J_form = ufl.derivative(R_form, u1, du)
    R_compiled = fem.form(R_form)
    J_compiled = fem.form(J_form)

    # CL de jauge / fixation numerique
    bcs = []
    if GAUGE_MODE == "face":
        fdim = domain.topology.dim - 1
        facets_fix = facet_tags.find(int(TAG_FIX_FACE))
        if facets_fix is None or len(facets_fix) == 0:
            available = "inconnu"
            try:
                available = ", ".join(str(int(v)) for v in np.unique(facet_tags.values))
            except Exception:
                pass
            raise ValueError(f"Tag de facettes introuvable: TAG_FIX_FACE={TAG_FIX_FACE}. Disponibles: {available}")
        dofs_fix = fem.locate_dofs_topological(V, fdim, facets_fix)
        u_fix = fem.Function(V)
        u_fix.x.array[:] = 0.0
        try:
            bcs = [fem.dirichletbc(u_fix, dofs_fix)]
        except TypeError:
            bcs = [fem.dirichletbc(u_fix, dofs_fix, V)]

    elif GAUGE_MODE == "minimal":
        # Jauge minimale "TD" pour enlever les 6 mouvements rigides sans bloquer une face complete:
        # A: ux=uy=uz, B: uy=uz, C: uz
        X = domain.geometry.x
        zmin = float(np.min(X[:, 2]))
        ztol = 1e-10 * max(1.0, float(np.max(np.abs(X[:, 2]))), 1.0) + 1e-12
        bottom_mask = np.abs(X[:, 2] - zmin) <= max(ztol, 1e-8)
        Xb = X[bottom_mask]
        if Xb.shape[0] < 3:
            raise RuntimeError("Impossible de construire la jauge minimale: pas assez de points sur z=zmin.")

        s1 = Xb[:, 0] + Xb[:, 1]
        s2 = Xb[:, 0] - Xb[:, 1]
        iA = int(np.argmin(s1))
        iB = int(np.argmax(s2))
        iC = int(np.argmax(s1))
        A_pt = Xb[iA].copy()
        B_pt = Xb[iB].copy()
        C_pt = Xb[iC].copy()

        pt_tol = 1e-10 + 1e-6 * max(float(np.max(X[:, 0]) - np.min(X[:, 0])), float(np.max(X[:, 1]) - np.min(X[:, 1])), 1.0)

        def marker_point(pt):
            return lambda x: (
                (np.abs(x[0] - pt[0]) <= pt_tol)
                & (np.abs(x[1] - pt[1]) <= pt_tol)
                & (np.abs(x[2] - pt[2]) <= pt_tol)
            )

        def locate_subspace_dofs(subspace, pt):
            marker = marker_point(pt)
            try:
                dofs = fem.locate_dofs_geometrical(subspace, marker)
            except Exception:
                Vsub_c, _map = subspace.collapse()
                dofs = fem.locate_dofs_geometrical((subspace, Vsub_c), marker)
            if len(dofs) == 0:
                raise RuntimeError(f"Aucun ddl trouve pour la jauge au point {pt}.")
            return dofs

        def bc_scalar_zero_on_point(comp, pt):
            Vsub = V.sub(comp)
            dofs = locate_subspace_dofs(Vsub, pt)
            try:
                return fem.dirichletbc(PETSc.ScalarType(0.0), dofs, Vsub)
            except Exception:
                # fallback old API: function on collapsed scalar space
                Vc, _map = Vsub.collapse()
                uc = fem.Function(Vc)
                uc.x.array[:] = 0.0
                return fem.dirichletbc(uc, dofs, Vsub)

        bcs = [
            bc_scalar_zero_on_point(0, A_pt),
            bc_scalar_zero_on_point(1, A_pt),
            bc_scalar_zero_on_point(2, A_pt),
            bc_scalar_zero_on_point(1, B_pt),
            bc_scalar_zero_on_point(2, B_pt),
            bc_scalar_zero_on_point(2, C_pt),
        ]
    else:
        raise ValueError("GAUGE_MODE doit etre 'minimal' ou 'face'.")

    # ------------------------------------------------------
    # Newton manuel (robuste entre versions dolfinx)
    # ------------------------------------------------------
    x_petsc = getattr(u1.x, "petsc_vec", None)
    if x_petsc is None:
        x_petsc = getattr(u1, "vector", None)
    if x_petsc is None:
        raise RuntimeError("Vecteur PETSc de u1 inaccessible.")

    ksp = PETSc.KSP().create(domain.comm)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setFromOptions()

    def assemble_residual():
        b = fem_petsc.assemble_vector(R_compiled)
        if bcs:
            fem_petsc.apply_lifting(b, [J_compiled], bcs=[bcs], x0=[x_petsc], alpha=-1.0)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem_petsc.set_bc(b, bcs, x_petsc, -1.0)
        else:
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        return b

    if domain.comm.rank == 0:
        print("=== Parametres ===")
        print("MESH_FILE =", MESH_FILE)
        print("TAG_LATERAL =", TAG_LATERAL)
        print("GAUGE_MODE =", GAUGE_MODE)
        print("TAG_FIX_FACE =", TAG_FIX_FACE)
        print("E =", E_val, "Pa")
        print("nu =", nu_val)
        print("rho0 =", rho0_val, "kg/m^3")
        print("mu =", mu, "Pa")
        print("kappa =", kappa, "Pa")
        print("T0_TARGET =", T0_TARGET, "Pa")
        print("N_LOAD_STEPS =", N_LOAD_STEPS)
        print("Jauge minimale recommandee pour ne presque pas alterer la physique du tambour.")

    load_values = np.linspace(0.0, float(T0_TARGET), max(1, int(N_LOAD_STEPS)) + 1)[1:]
    for i_load, Tload in enumerate(load_values, start=1):
        T0.value = PETSc.ScalarType(Tload)
        if domain.comm.rank == 0:
            print(f"\n-- Load step {i_load}/{len(load_values)} : T = {Tload:.6e} Pa")

        converged = False
        res0 = None
        for k_newton in range(int(NEWTON_MAX_IT)):
            b = assemble_residual()
            res = float(b.norm())
            if res0 is None:
                res0 = max(res, 1e-30)
            rel = res / res0
            if domain.comm.rank == 0:
                print(f"Newton iter {k_newton:02d}: ||R|| = {res:.6e} (rel={rel:.3e})")

            if (res <= NEWTON_ATOL) or (rel <= NEWTON_RTOL):
                converged = True
                b.destroy()
                break

            A = fem_petsc.assemble_matrix(J_compiled, bcs=bcs)
            A.assemble()

            dx = b.duplicate()
            dx.set(0.0)
            b.scale(-1.0)
            ksp.setOperators(A)
            ksp.solve(b, dx)

            x_old = x_petsc.duplicate()
            x_old.copy(x_petsc)

            alpha = 1.0
            accepted = False
            best_alpha = 1.0
            best_res = None
            for _ in range(12):
                x_petsc.copy(x_old)
                x_petsc.axpy(alpha, dx)
                x_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

                b_trial = assemble_residual()
                r_trial = float(b_trial.norm())
                b_trial.destroy()

                if best_res is None or r_trial < best_res:
                    best_res = r_trial
                    best_alpha = alpha

                # version simple (style TD): accepter toute baisse
                if r_trial < res:
                    accepted = True
                    break
                alpha *= 0.5

            # fallback: si le meilleur essai ne degrade presque pas, on accepte
            if (not accepted) and (best_res is not None) and (best_res <= 1.05 * res):
                x_petsc.copy(x_old)
                x_petsc.axpy(best_alpha, dx)
                x_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
                accepted = True

            # Si la line search echoue mais qu'on est deja tres proche, on s'arrete.
            # (Evite de repartir loin de la solution en acceptant un mauvais pas.)
            if (not accepted) and ((res <= 10.0 * NEWTON_ATOL) or (rel <= 2.0 * NEWTON_RTOL)):
                converged = True
                if domain.comm.rank == 0:
                    print("  warning: line search echoue pres de la convergence -> on accepte la convergence.")
                x_petsc.copy(x_old)
                x_petsc.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            step_norm = float(dx.norm())

            x_old.destroy()
            dx.destroy()
            A.destroy()
            b.destroy()

            if (not accepted) and (not converged):
                # Si le residu est deja faible relativement, on accepte et on sort
                if rel <= max(5.0 * NEWTON_RTOL, 1e-4):
                    converged = True
                    break
                raise RuntimeError(f"Line search echoue (load step {i_load}, iter {k_newton}).")

            if converged:
                break

            if step_norm < 1e-14:
                converged = True
                break

        if not converged:
            raise RuntimeError(f"Prechargement non convergent au load step {i_load}/{len(load_values)}")

    u1.x.scatter_forward()

    # ------------------------------------------------------
    # Vibrations lineaires autour de l'etat precontraint
    # ------------------------------------------------------
    xi = ufl.TrialFunction(V)
    eta = ufl.TestFunction(V)

    F_ref = I + ufl.grad(u1)
    F_ref_var = ufl.variable(F_ref)
    C_ref = F_ref_var.T * F_ref_var
    J_ref = ufl.det(F_ref_var)
    I1_ref = ufl.tr(C_ref)
    I1_bar_ref = J_ref ** (-2.0 / 3.0) * I1_ref
    psi_ref = 0.5 * mu * (I1_bar_ref - 3.0) + 0.25 * kappa * (J_ref * J_ref - 1.0 - 2.0 * ufl.ln(J_ref))
    P_ref = ufl.diff(psi_ref, F_ref_var)

    internal = ufl.inner(P_ref, ufl.grad(eta)) * ufl.dx
    a_form = ufl.derivative(internal, u1, xi)  # tangente totale
    m_form = float(rho0_val) * ufl.inner(xi, eta) * ufl.dx

    K = fem_petsc.assemble_matrix(fem.form(a_form), bcs=bcs)
    K.assemble()
    M = fem_petsc.assemble_matrix(fem.form(m_form), bcs=bcs)
    M.assemble()

    # ------------------------------------------------------
    # Probleme modal : K phi = omega^2 M phi
    # ------------------------------------------------------
    eps = SLEPc.EPS().create(domain.comm)
    eps.setOperators(K, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev=int(N_MODES))
    eps.setTarget(float(EIG_TARGET))
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    eps.getST().setType(SLEPc.ST.Type.SINVERT)
    eps.setFromOptions()
    eps.solve()

    vr, vi = K.createVecs()
    eigpairs = []
    for k in range(eps.getConverged()):
        eig = eps.getEigenpair(k, vr, vi)
        lam = float(np.real(eig))
        lam_im = float(np.imag(eig))
        if abs(lam_im) > 1e-10:
            continue
        if lam <= 1e-12:
            continue
        eigpairs.append((lam, vr.getArray(readonly=True).copy()))
    eigpairs.sort(key=lambda p: p[0])
    eigpairs = eigpairs[: int(N_MODES)]

    if domain.comm.rank == 0:
        print("\n=== Modes propres ===")
        if not eigpairs:
            print("Aucun mode positif detecte.")
        for i, (lam, _) in enumerate(eigpairs, start=1):
            omega = np.sqrt(lam)
            print(f"mode {i:02d}: omega^2={lam:.6e}, omega={omega:.6e} rad/s, f={omega/(2*np.pi):.6f} Hz")

    # ------------------------------------------------------
    # Exports (fem/results)
    # ------------------------------------------------------
    results_dir = Path("fem/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_BASENAME if RESULTS_BASENAME else Path(MESH_FILE).stem
    csv_path = results_dir / f"{stem}_modes.csv"
    vtk_path = results_dir / f"{stem}_modes.pvd"

    if WRITE_CSV:
        with csv_path.open("w", newline="") as f:
            import csv

            writer = csv.writer(f)
            writer.writerow(["mode_index", "omega2", "omega_rad_s", "f_Hz"])
            for i, (lam, _) in enumerate(eigpairs, start=1):
                omega = float(np.sqrt(lam))
                writer.writerow([i, lam, omega, omega / (2.0 * np.pi)])
        if domain.comm.rank == 0:
            print("CSV exporte:", csv_path)

    if WRITE_VTK:
        mode_fun = fem.Function(V, name="mode")
        u_out = fem.Function(V, name="u_prestress")
        u_out.x.array[:] = u1.x.array
        u_out.x.scatter_forward()

        with io.VTKFile(domain.comm, str(vtk_path), "w") as vtk:
            vtk.write_mesh(domain)
            vtk.write_function(u_out, 0.0)
            for i, (lam, vec) in enumerate(eigpairs, start=1):
                mode_fun.x.array[:] = vec
                mode_fun.x.scatter_forward()
                mode_fun.name = f"mode_{i}"
                vtk.write_function(mode_fun, float(lam))
        if domain.comm.rank == 0:
            print("VTK exporte:", vtk_path)


if __name__ == "__main__":
    main()
