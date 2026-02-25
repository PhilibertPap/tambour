# Projet MEC570 - Tambour

## Objectif du projet

Ce projet contient un calcul par éléments finis pour un tambour de géométrie quelconque :

- précontrainte hyperélastique en grandes transformations (quasi-statique),
- puis calcul des modes propres autour de l'état précontraint (petites vibrations).

## Structure du projet

- `fem/main.py`
  - script principal FEniCSx
  - résout la précontrainte puis le problème modal
  - exporte les fréquences (CSV) et les champs (`.pvd/.pvtu/.vtu`) pour ParaView

- `fem/mesh_cercle.py`, `fem/mesh_ellipse.py`, `fem/mesh_carre.py`
  - scripts Gmsh simples pour générer des maillages 3D
  - mêmes tags physiques pour tous les cas

- `fem/mesh/`
  - stockage des maillages

- `fem/results/`
  - stockage des résultats

- notebooks de TD (`fem/E8...`, `fem/E9.1...`, `fem/E9.2...`)
  - utilisés comme référence de style et de démarche

## Convention de tags physiques (Gmsh)

Les scripts de maillage (`fem/mesh_*.py`) utilisent les mêmes tags :

- `1` : volume (`Omega0`)
- `10` : bord latéral (`Gamma_lat`)
- `11` : face supérieure (`Gamma_plus`)
- `12` : face inférieure (`Gamma_minus`)

## Workflow recommandé (simple)

### 1. Générer un maillage

Choisir un script de maillage et modifier ses constantes en haut du fichier, puis exécuter :

```bash
python fem/mesh_cercle.py
```

Exemple de sortie :

- `fem/mesh/cercle/cercle.msh`

### 2. Lancer le calcul FEM

Modifier les constantes en haut de `fem/main.py` (maillage, matériau, traction, nombre de modes, etc.), puis exécuter :

```bash
python fem/main.py
```

### 3. Visualiser les résultats dans ParaView

Ouvrir le fichier :

- `fem/mesh/<cas>/results/<cas>_modes.pvd`

Puis appliquer éventuellement `Warp By Vector` pour mieux visualiser les modes.

## Notes numériques

- Le script utilise une **continuation en charge** (`N_LOAD_STEPS`) pour améliorer la robustesse du préchargement non linéaire.
- Une **jauge minimale** (`GAUGE_MODE = "minimal"`) est utilisée pour supprimer les mouvements de corps rigide sans bloquer une face entière.
- Pour des tests rapides, commencer avec :
  - maillage plus grossier (`LC` plus grand),
  - traction plus faible (`T0_TARGET` plus petit),
  - puis raffiner progressivement.

## Bonnes pratiques Git

Les fichiers de résultats générés (VTK/CSV) sont ignorés via `.gitignore`, ce qui évite d'encombrer l'historique Git avec des sorties volumineuses.
