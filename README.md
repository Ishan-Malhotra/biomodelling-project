# Biomolecular coordinate converter

Convert a protein backbone between Cartesian coordinates (x, y, z in Ångström) and internal coordinates (φ, ψ, ω, plus the carbonyl oxygen). The same conversion code runs in a terminal program and on the website.

Live site: [biomodelling-converter.vercel.app](https://biomodelling-converter.vercel.app)

Ishan Malhotra (2023B1A30837P)

## What it converts

Cartesian coordinates say where each atom is. Internal coordinates say what shape the chain is. For each residue the tool reports:

| Quantity | Meaning |
|---|---|
| φ (phi) | C(i−1)–N–CA–C |
| ψ (psi) | N–CA–C–N(i+1) |
| ω (omega) | CA–C–N(i+1)–CA(i+1) |
| C=O bond | carbonyl bond length |
| CA–C–O | carbonyl bond angle |
| N–CA–C–O | dihedral that places the oxygen |

The first residue has no φ, and the last residue has no ω, because those angles need atoms from a neighbouring residue. A blank angle is left blank. The program does not fill in helix values, and it will not join separate chains or a chain break into one peptide.

Dihedral signs follow the IUPAC convention (positive is clockwise when looking along the middle bond). The range is −180° to 180°, and 180° and −180° are the same conformation.

Building a backbone from φ, ψ and ω uses ideal geometry from Engh and Huber (1991): N–CA 1.458 Å, CA–C 1.525 Å, C–N 1.329 Å, C=O 1.231 Å; N–CA–C 111.2°, CA–C–N 116.2°, C–N–CA 121.7°, CA–C–O 120.5°. If the oxygen dihedral is left blank, it is placed in the peptide plane as ψ + 180°. Atoms are placed with the NeRF method (Parsons et al., 2005).

Presets: α-helix (−57°, −47°, 180°), β-strand (−139°, 135°, 180°), polyproline II (−75°, 145°, 180°) and extended (180°, 180°, 180°).

## Run the terminal program

Python 3.9 or newer. The maths and the terminal program use only the standard library.

```bash
python3 main.py
```

Choose Cartesian or internal coordinates, then type atoms, load a PDB, paste a residue table, or build a secondary-structure preset. A successful conversion prints a check that the rebuilt structure matches the input. Results can be saved as PDB or CSV under `output/`.

## Run the website locally

```bash
python3 -m venv .venv
.venv/bin/pip install fastapi uvicorn
.venv/bin/uvicorn web.server:app --reload
```

Open http://127.0.0.1:8000. The page is the static files in `public/`; the API is the FastAPI app in `web/server.py`.

The site is deployed on Vercel from this repository (`pyproject.toml` sets the entrypoint to `web.server:app`).

## Tests

```bash
python3 -m unittest discover -s tests -q
```

A conversion is shown only when the rebuild matches. Distances must agree to 10⁻⁶ Å and angles to 10⁻⁴°. The check uses signed internal coordinates, so a mirror image is rejected. Structures larger than 1500 atoms skip the all-pairs distance check and use the internal-coordinate check instead.

## Layout

```
main.py                 terminal program
coordconv/geometry.py   distances, angles, dihedrals
coordconv/conversions.py  Z-matrix, backbone builder, match checks
coordconv/io_utils.py   PDB and residue CSV
web/server.py           API
public/                 website
settings.py             name, ID and structure type shown on the start screen
tests/
```

## Limits

The builder writes backbone atoms only: N, CA, C and O. It does not add side chains or hydrogens. PDB input uses ATOM records from the first model (alternate location blank or A). Residue numbers must fit in a PDB column (−999 to 9999), and an insertion code is kept (27 and 27A stay distinct). The helix and strand labels are approximate Ramachandran regions, not a full secondary-structure assignment.
