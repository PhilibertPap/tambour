# Traction Drumhead (Neo-Hookean)

Ce script résout un essai de traction axisymétrique avec un modèle `NeoHookeanReducedI1`.

## Entrées matériau

Trois façons de définir le matériau:

1. `--mu` et `--K` directement (Pa)
2. `--E` et `--nu` (conversion automatique en `mu`, `K`)
3. `--preset` matériau:
   - `mylar_92` (film PET/Mylar, ordre de grandeur)
   - `steel_304` (inox 304)

## Presets de charge (tambour)

Le script utilise des presets de charge **en fraction de `mu`**:

- `drum_classic_relax`
  - `T_max = 0.10*mu`
  - réponse souvent proche du linéaire
- `drum_classic_tendu` (recommandé par défaut)
  - `T_max = 0.35*mu`
  - non-linéarité modérée, visible
- `drum_classic_tres_tendu`
  - `T_max = 0.70*mu`
  - non-linéarité marquée

Pas de continuation associés:

- `drum_classic_relax`: `dT0=0.005*mu`, `dT_max=0.02*mu`
- `drum_classic_tendu`: `dT0=0.015*mu`, `dT_max=0.05*mu`
- `drum_classic_tres_tendu`: `dT0=0.03*mu`, `dT_max=0.10*mu`

## Sorties graphiques

Le script ouvre **deux figures séparées**:

1. `lambda_r` et `lambda_z` vs `T/mu`
2. `J` vs `T/mu`

`J` est séparé car sa signification/échelle est différente des stretches.

## Exemples

Cas par défaut (tambour classique tendu):

```bash
python run_continuation.py
```

Même cas mais matériau Mylar preset:

```bash
python run_continuation.py --preset mylar_92
```

Cas plus non-linéaire:

```bash
python run_continuation.py --load-preset drum_classic_tres_tendu
```

Cas custom total:

```bash
python run_continuation.py --mu 1.2e9 --K 4.5e9 --T-max 5e8 --dT0 2e7 --dT-max 8e7
```

## Interprétation rapide

- Si `T_max/mu < 0.1`: réponse souvent quasi-linéaire.
- Si `T_max/mu ~ 0.3 - 0.8`: non-linéarité généralement visible.
