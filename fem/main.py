from __future__ import annotations

from pathlib import Path

import numpy as np


# ==========================================================
# PARAMETRES
# ==========================================================

# Maillage / tags (les scripts dans fem/mesh utilisent ces tags)
MESH_FILE = "fem/mesh/cercle/cercle.msh"
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

# Precontrainte
# "imposed_radial_displacement" (recommande pour un tambour serre)
# "traction" (cas d'etude)
PRESTRESS_MODE = "imposed_radial_displacement"
T0_TARGET = 1.0e6                    # Pa (si PRESTRESS_MODE == "traction")
EDGE_RADIAL_DISPLACEMENT_TARGET = 2e-4  # m (si PRESTRESS_MODE == "imposed_radial_displacement")
N_LOAD_STEPS = 10                    # continuation simple

# Discretisation / solveurs
FE_DEGREE = 2
NEWTON_ATOL = 1e-8
NEWTON_RTOL = 5e-4   # tolerance relative par pas de charge
NEWTON_MAX_IT = 40
N_MODES = 10
EIG_TARGET = 0.0

# Exports
WRITE_CSV = True
WRITE_VTK = True
RESULTS_BASENAME = None  # ex: "cercle_T1e6"; None => derive du nom de maillage
NORMALIZE_MODES_FOR_VTK = True  # rend les modes visibles dans ParaView
VTK_MODE_SCALE = 1.0            # facteur supplementaire (ex: 10.0)


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

    if PRESTRESS_MODE == "traction":
        R_form = ufl.inner(P1, ufl.grad(v)) * ufl.dx - ufl.inner(traction, v) * ds(TAG_LATERAL)
    elif PRESTRESS_MODE == "imposed_radial_displacement":
        R_form = ufl.inner(P1, ufl.grad(v)) * ufl.dx
    else:
        raise ValueError("PRESTRESS_MODE doit etre 'traction' ou 'imposed_radial_displacement'.")
    J_form = ufl.derivative(R_form, u1, du)

    # CL de prechargement + CL des vibrations
    # - prechargement:
    #     * "imposed_radial_displacement" -> serrage cinematique sur Gamma_lat
    #     * "traction" -> traction sur Gamma_lat + jauge minimale/face
    # - vibrations:
    #     * bord lateral bloque (tambour serre)
    bcs_prestress = []
    if PRESTRESS_MODE == "imposed_radial_displacement":
        fdim = domain.topology.dim - 1
        facets_lat = facet_tags.find(int(TAG_LATERAL))
        if facets_lat is None or len(facets_lat) == 0:
            raise ValueError(f"Tag lateral introuvable: TAG_LATERAL={TAG_LATERAL}")
        dofs_lat = fem.locate_dofs_topological(V, fdim, facets_lat)

        u_lat = fem.Function(V, name="u_lateral_bc")

        def update_lateral_bc(amplitude):
            # Serrage simple "TD": deplacement radial dans le plan (x,y), uz=0 sur le bord
            amp = float(amplitude)

            def expr(x):
                values = np.zeros((3, x.shape[1]), dtype=np.float64)
                values[0, :] = amp * x[0, :]
                values[1, :] = amp * x[1, :]
                values[2, :] = 0.0
                return values

            u_lat.interpolate(expr)
            u_lat.x.scatter_forward()

        update_lateral_bc(0.0)
        try:
            bcs_prestress = [fem.dirichletbc(u_lat, dofs_lat)]
        except TypeError:
            bcs_prestress = [fem.dirichletbc(u_lat, dofs_lat, V)]

    elif GAUGE_MODE == "face":
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
            bcs_prestress = [fem.dirichletbc(u_fix, dofs_fix)]
        except TypeError:
            bcs_prestress = [fem.dirichletbc(u_fix, dofs_fix, V)]

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

        bcs_prestress = [
            bc_scalar_zero_on_point(0, A_pt),
            bc_scalar_zero_on_point(1, A_pt),
            bc_scalar_zero_on_point(2, A_pt),
            bc_scalar_zero_on_point(1, B_pt),
            bc_scalar_zero_on_point(2, B_pt),
            bc_scalar_zero_on_point(2, C_pt),
        ]
    else:
        raise ValueError("GAUGE_MODE doit etre 'minimal' ou 'face'.")

    # Vibrations: bord lateral bloque (perturbation nulle)
    fdim = domain.topology.dim - 1
    facets_lat = facet_tags.find(int(TAG_LATERAL))
    if facets_lat is None or len(facets_lat) == 0:
        raise ValueError(f"Tag lateral introuvable: TAG_LATERAL={TAG_LATERAL}")
    dofs_lat_modes = fem.locate_dofs_topological(V, fdim, facets_lat)
    u_zero_mode = fem.Function(V)
    u_zero_mode.x.array[:] = 0.0
    try:
        bcs_modes = [fem.dirichletbc(u_zero_mode, dofs_lat_modes)]
    except TypeError:
        bcs_modes = [fem.dirichletbc(u_zero_mode, dofs_lat_modes, V)]

    # ------------------------------------------------------
    # Solveur quasi-statique non lineaire (SNES/PETSc)
    # style notebook: definir F, J, puis resoudre a chaque pas de charge
    # ------------------------------------------------------
    problem = fem_petsc.NonlinearProblem(
        R_form,
        u1,
        bcs=bcs_prestress,
        J=J_form,
        petsc_options_prefix="prestress_",
        petsc_options={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "snes_max_it": int(NEWTON_MAX_IT),
            "snes_atol": float(NEWTON_ATOL),
            "snes_rtol": float(NEWTON_RTOL),
            "snes_stol": 1e-14,
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )

    if domain.comm.rank == 0:
        print("=== Parametres ===")
        print("MESH_FILE =", MESH_FILE)
        print("TAG_LATERAL =", TAG_LATERAL)
        print("GAUGE_MODE =", GAUGE_MODE)
        print("TAG_FIX_FACE =", TAG_FIX_FACE)
        print("PRESTRESS_MODE =", PRESTRESS_MODE)
        print("E =", E_val, "Pa")
        print("nu =", nu_val)
        print("rho0 =", rho0_val, "kg/m^3")
        print("mu =", mu, "Pa")
        print("kappa =", kappa, "Pa")
        print("T0_TARGET =", T0_TARGET, "Pa")
        print("EDGE_RADIAL_DISPLACEMENT_TARGET =", EDGE_RADIAL_DISPLACEMENT_TARGET, "m")
        print("N_LOAD_STEPS =", N_LOAD_STEPS)
        print("Jauge minimale recommandee pour ne presque pas alterer la physique du tambour.")
        print("NORMALIZE_MODES_FOR_VTK =", NORMALIZE_MODES_FOR_VTK)

    if PRESTRESS_MODE == "traction":
        load_values = np.linspace(0.0, float(T0_TARGET), max(1, int(N_LOAD_STEPS)) + 1)[1:]
    else:
        load_values = np.linspace(0.0, float(EDGE_RADIAL_DISPLACEMENT_TARGET), max(1, int(N_LOAD_STEPS)) + 1)[1:]

    for i_load, load_val in enumerate(load_values, start=1):
        if PRESTRESS_MODE == "traction":
            T0.value = PETSc.ScalarType(load_val)
        else:
            update_lateral_bc(float(load_val))
        if domain.comm.rank == 0:
            if PRESTRESS_MODE == "traction":
                print(f"\n-- Load step {i_load}/{len(load_values)} : T = {load_val:.6e} Pa")
            else:
                print(f"\n-- Load step {i_load}/{len(load_values)} : u_edge = {load_val:.6e} m")
        problem.solve()
        reason = problem.solver.getConvergedReason()
        its = problem.solver.getIterationNumber()
        u1.x.scatter_forward()
        if domain.comm.rank == 0:
            print(f"SNES iters = {its}, reason = {reason}")
        if reason <= 0:
            raise RuntimeError(f"SNES non convergent au load step {i_load}/{len(load_values)} (reason={reason}).")

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

    K = fem_petsc.assemble_matrix(fem.form(a_form), bcs=bcs_modes)
    K.assemble()
    M = fem_petsc.assemble_matrix(fem.form(m_form), bcs=bcs_modes)
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
    # Exports (dans fem/results/<cas>/)
    # ------------------------------------------------------
    mesh_path = Path(MESH_FILE)
    case_name = mesh_path.parent.name if mesh_path.parent.name != "mesh" else mesh_path.stem
    results_dir = Path("fem/results") / case_name
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_BASENAME if RESULTS_BASENAME else mesh_path.stem
    csv_path = results_dir / f"{stem}_modes.csv"
    vtk_prestress_path = results_dir / f"{stem}_prestress.pvd"
    vtk_modes_path = results_dir / f"{stem}_modes.pvd"

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

        # 1) Prechargement seul (plus simple a visualiser)
        with io.VTKFile(domain.comm, str(vtk_prestress_path), "w") as vtk:
            vtk.write_mesh(domain)
            vtk.write_function(u_out, 0.0)

        # 2) Modes seuls, ecrits comme serie temporelle (un mode par "temps")
        with io.VTKFile(domain.comm, str(vtk_modes_path), "w") as vtk:
            vtk.write_mesh(domain)
            # Champ nul au temps 0 pour que ParaView voie immediatement un vecteur "mode"
            mode_fun.x.array[:] = 0.0
            mode_fun.x.scatter_forward()
            mode_fun.name = "mode"
            vtk.write_function(mode_fun, 0.0)
            for i, (lam, vec) in enumerate(eigpairs, start=1):
                vec_out = vec.copy()
                if NORMALIZE_MODES_FOR_VTK:
                    vmax = float(np.max(np.abs(vec_out)))
                    if vmax > 0.0:
                        vec_out /= vmax
                vec_out *= float(VTK_MODE_SCALE)

                mode_fun.x.array[:] = vec_out
                mode_fun.x.scatter_forward()
                mode_fun.name = "mode"  # meme nom de champ, indexe par le temps
                vtk.write_function(mode_fun, float(i))
        if domain.comm.rank == 0:
            print("VTK prechargement exporte:", vtk_prestress_path)
            print("VTK modes exporte:", vtk_modes_path)


if __name__ == "__main__":
    main()
