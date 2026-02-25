from __future__ import annotations

from pathlib import Path

import gmsh


# ==========================================================
# PARAMETRES (modifier ici puis executer)
# ==========================================================
# Exemple "caisse claire" courant: 14" de diametre -> R = 0.1778 m
RADIUS = 0.1778
# Peau batter simple pli ~10 mil ~= 0.254 mm (ordre de grandeur realiste)
THICKNESS = 2.54e-4
LC = 0.015
OUT_FILE = "fem/mesh/cercle/cercle.msh"

# Tags physiques (utilises par fem/main.py)
VOLUME_TAG = 1
LATERAL_TAG = 10
TOP_TAG = 11
BOTTOM_TAG = 12


def main():
    out = Path(OUT_FILE)
    out.parent.mkdir(parents=True, exist_ok=True)

    H = float(THICKNESS)
    z0 = -0.5 * H

    gmsh.initialize()
    gmsh.model.add("tambour_cercle")
    occ = gmsh.model.occ

    disk = occ.addDisk(0.0, 0.0, z0, float(RADIUS), float(RADIUS))
    ext = occ.extrude([(2, disk)], 0.0, 0.0, H)
    occ.synchronize()

    vol_tags = [tag for dim, tag in ext if dim == 3]
    surf_tags = [disk] + [tag for dim, tag in ext if dim == 2]

    top, bottom, lateral = [], [], []
    tol = 1e-9 * max(1.0, H)
    for s in surf_tags:
        _, _, z = occ.getCenterOfMass(2, s)
        if abs(z - (z0 + H)) < tol:
            top.append(s)
        elif abs(z - z0) < tol:
            bottom.append(s)
        else:
            lateral.append(s)

    gmsh.model.addPhysicalGroup(3, vol_tags, VOLUME_TAG)
    gmsh.model.setPhysicalName(3, VOLUME_TAG, "Omega0")
    gmsh.model.addPhysicalGroup(2, lateral, LATERAL_TAG)
    gmsh.model.setPhysicalName(2, LATERAL_TAG, "Gamma_lat")
    gmsh.model.addPhysicalGroup(2, top, TOP_TAG)
    gmsh.model.setPhysicalName(2, TOP_TAG, "Gamma_plus")
    gmsh.model.addPhysicalGroup(2, bottom, BOTTOM_TAG)
    gmsh.model.setPhysicalName(2, BOTTOM_TAG, "Gamma_minus")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(LC))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(LC))
    gmsh.model.mesh.generate(3)
    gmsh.write(str(out))

    print("Maillage ecrit:", out)
    print("Tags: VOLUME=1, LATERAL=10, TOP=11, BOTTOM=12")
    gmsh.finalize()


if __name__ == "__main__":
    main()
