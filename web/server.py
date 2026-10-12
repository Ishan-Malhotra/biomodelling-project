"""Web API for the coordinate converter (deployed to Vercel as a FastAPI app).

Local development:
    pip install fastapi uvicorn
    uvicorn web.server:app --reload
"""

import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import settings
from coordconv import conversions as conv
from coordconv import io_utils as io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(ROOT, "examples")
PUBLIC_DIR = os.path.join(ROOT, "public")

MAX_ATOMS = 10000
# The all-pairs distance check is O(n^2), so larger structures use a cheaper check.
DISTANCE_CHECK_LIMIT = 1500

EXAMPLES = {
    "cartesian": "crambin_fragment.pdb",
    "protein": "1crn.pdb",
    "zmatrix": "crambin_fragment.zmat",
}

app = FastAPI(title="Biomolecular coordinate converter")


class CartesianRequest(BaseModel):
    text: str
    backbone_only: bool = False


class ZMatrixRequest(BaseModel):
    text: str


class ResidueTorsionInput(BaseModel):
    phi: Optional[float] = None
    psi: Optional[float] = None
    omega: Optional[float] = None


class BackboneRequest(BaseModel):
    count: int = Field(ge=1, le=500)
    sequence: str = ""
    preset: Optional[str] = None
    residues: Optional[List[ResidueTorsionInput]] = None


def bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def atom_json(atom: conv.Atom) -> dict:
    x, y, z = atom.xyz
    return {"name": atom.name, "residue": "{}{}".format(atom.res_name, atom.res_seq),
            "chain": atom.chain, "x": x, "y": y, "z": z}


def zentry_json(entry: conv.ZEntry) -> dict:
    def ref(index):
        return None if index is None else index + 1

    return {"name": entry.name, "bond_ref": ref(entry.bond_ref), "bond": entry.bond,
            "angle_ref": ref(entry.angle_ref), "angle": entry.angle,
            "dihedral_ref": ref(entry.dihedral_ref), "dihedral": entry.dihedral}


def torsions_json(atoms: List[conv.Atom]) -> List[dict]:
    torsions = conv.backbone_torsions(atoms)
    if not any(t.phi is not None or t.psi is not None for t in torsions):
        return []
    return [{"residue": "{}{}".format(t.res_name, t.res_seq), "chain": t.chain,
             "phi": t.phi, "psi": t.psi, "omega": t.omega,
             "region": conv.ramachandran_region(t.phi, t.psi)} for t in torsions]


def bonds_json(entries: List[conv.ZEntry]) -> List[List[int]]:
    return [[i, e.bond_ref] for i, e in enumerate(entries) if e.bond_ref is not None]


def internal_check(entries, coords) -> dict:
    bond, angle, dihedral = conv.zmatrix_errors(entries, coords)
    return {"kind": "internal", "bond": bond, "angle": angle, "dihedral": dihedral}


def credit(action: str) -> str:
    return "{} by {} ({})".format(action, settings.STUDENT_NAME, settings.STUDENT_ID)


@app.get("/api/profile")
def profile():
    return {"name": settings.STUDENT_NAME, "id": settings.STUDENT_ID,
            "structure": settings.STRUCTURE_TYPE,
            "presets": {k: list(v) for k, v in conv.SECONDARY_STRUCTURE_PRESETS.items()}}


@app.get("/api/examples/{name}")
def example(name: str):
    if name not in EXAMPLES:
        raise HTTPException(status_code=404, detail="unknown example '{}'".format(name))
    with open(os.path.join(EXAMPLES_DIR, EXAMPLES[name])) as fh:
        return {"name": EXAMPLES[name], "text": fh.read()}


@app.post("/api/cartesian-to-internal")
def cartesian_to_internal(req: CartesianRequest):
    try:
        atoms = io.parse_cartesian_text(req.text, req.backbone_only)
        if len(atoms) < 2:
            raise ValueError("enter at least 2 atoms - internal coordinates start with a bond")
        if len(atoms) > MAX_ATOMS:
            raise ValueError("the web version handles up to {} atoms; use the terminal app for larger structures".format(MAX_ATOMS))
        entries = conv.cartesian_to_zmatrix(atoms)
        rebuilt = conv.zmatrix_to_cartesian(entries)
    except ValueError as exc:
        raise bad_request(exc) from None

    if len(atoms) <= DISTANCE_CHECK_LIMIT:
        check = {"kind": "distances", "pairs": len(atoms) * (len(atoms) - 1) // 2,
                 "max_deviation": conv.max_distance_deviation([a.xyz for a in atoms], rebuilt)}
    else:
        check = internal_check(entries, rebuilt)

    return {
        "atoms": [atom_json(a) for a in atoms],
        "zmatrix": [zentry_json(e) for e in entries],
        "torsions": torsions_json(atoms),
        "bonds": bonds_json(entries),
        "check": check,
        "download": {"filename": "zmatrix.zmat", "text": io.zmatrix_text(entries, credit("Z-matrix generated"))},
    }


@app.post("/api/internal-to-cartesian")
def internal_to_cartesian(req: ZMatrixRequest):
    try:
        entries = io.parse_zmatrix_text(req.text)
        if len(entries) > MAX_ATOMS:
            raise ValueError("the web version handles up to {} atoms".format(MAX_ATOMS))
        coords = conv.zmatrix_to_cartesian(entries)
    except ValueError as exc:
        raise bad_request(exc) from None

    atoms = io.assign_residues([conv.Atom(e.name, xyz) for e, xyz in zip(entries, coords)])
    return {
        "atoms": [atom_json(a) for a in atoms],
        "zmatrix": [zentry_json(e) for e in entries],
        "torsions": torsions_json(atoms),
        "bonds": bonds_json(entries),
        "check": internal_check(entries, coords),
        "download": {"filename": "structure.pdb", "text": io.pdb_text(atoms, credit("Built from a Z-matrix"))},
    }


@app.post("/api/backbone-to-cartesian")
def backbone_to_cartesian(req: BackboneRequest):
    try:
        res_names = io.sequence_to_residues(req.sequence, req.count)
        if req.preset:
            if req.preset not in conv.SECONDARY_STRUCTURE_PRESETS:
                raise ValueError("unknown conformation '{}'".format(req.preset))
            phi, psi, omega = conv.SECONDARY_STRUCTURE_PRESETS[req.preset]
            phis, psis, omegas = [phi] * req.count, [psi] * req.count, [omega] * req.count
        else:
            residues = req.residues or []
            if len(residues) != req.count:
                raise ValueError("expected torsions for {} residues, got {}".format(req.count, len(residues)))
            phis = [r.phi for r in residues]
            psis = [r.psi for r in residues]
            omegas = [r.omega for r in residues]
            for i in range(req.count):
                needed = ([("phi", phis[i])] if i > 0 else []) + (
                    [("psi", psis[i]), ("omega", omegas[i])] if i < req.count - 1 else [])
                missing = [name for name, value in needed if value is None]
                if missing:
                    raise ValueError("residue {} is missing {}".format(i + 1, " and ".join(missing)))
        phis[0] = None
        psis[-1] = None
        omegas[-1] = None
        entries = conv.build_backbone_zmatrix(res_names, phis, psis, omegas)
        atoms = conv.build_backbone(res_names, phis, psis, omegas)
    except ValueError as exc:
        raise bad_request(exc) from None

    return {
        "atoms": [atom_json(a) for a in atoms],
        "zmatrix": [zentry_json(e) for e in entries],
        "torsions": torsions_json(atoms),
        "bonds": bonds_json(entries),
        "check": {"kind": "torsions", "max_error": conv.backbone_torsion_error(atoms, phis, psis, omegas)},
        "download": {"filename": "backbone.pdb", "text": io.pdb_text(atoms, credit("Backbone built"))},
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))


# On Vercel the public/ folder is served by the CDN; locally FastAPI serves it.
if not os.environ.get("VERCEL"):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR), name="public")
