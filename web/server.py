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
PUBLIC_DIR = os.path.join(ROOT, "public")

MAX_ATOMS = 10000
MAX_RESIDUES = 500
# The all-pairs distance check is O(n^2), so larger structures use a cheaper check.
DISTANCE_CHECK_LIMIT = 1500

app = FastAPI(title="Biomolecular coordinate converter")


class CartesianRequest(BaseModel):
    text: str
    backbone_only: bool = False


class ZMatrixRequest(BaseModel):
    text: str


class ResidueInput(BaseModel):
    res_name: Optional[str] = None
    res_seq: Optional[int] = None
    chain: Optional[str] = None
    icode: Optional[str] = None
    phi: Optional[float] = None
    psi: Optional[float] = None
    omega: Optional[float] = None
    o_bond: Optional[float] = None
    o_angle: Optional[float] = None
    o_dihedral: Optional[float] = None


class BackboneRequest(BaseModel):
    count: Optional[int] = Field(default=None, ge=1, le=MAX_RESIDUES)
    sequence: str = ""
    preset: Optional[str] = None
    residues: Optional[List[ResidueInput]] = Field(default=None, max_length=MAX_RESIDUES)


def bad_request(exc: Exception) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def atom_json(atom: conv.Atom) -> dict:
    x, y, z = atom.xyz
    return {"name": atom.name, "residue": io.residue_id(atom.res_name, atom.res_seq, atom.icode),
            "chain": atom.chain, "x": x, "y": y, "z": z}


def zentry_json(entry: conv.ZEntry) -> dict:
    def ref(index):
        return None if index is None else index + 1

    return {"name": entry.name, "bond_ref": ref(entry.bond_ref), "bond": entry.bond,
            "angle_ref": ref(entry.angle_ref), "angle": entry.angle,
            "dihedral_ref": ref(entry.dihedral_ref), "dihedral": entry.dihedral}


def residue_torsions(atoms: List[conv.Atom]) -> List[conv.ResidueTorsions]:
    torsions = conv.backbone_torsions(atoms)
    if not any(t.phi is not None or t.psi is not None or t.o_dihedral is not None for t in torsions):
        return []
    return torsions


def torsions_json(torsions: List[conv.ResidueTorsions]) -> List[dict]:
    return [{"residue": io.residue_id(t.res_name, t.res_seq, t.icode), "chain": t.chain,
             "phi": t.phi, "psi": t.psi, "omega": t.omega,
             "o_bond": t.o_bond, "o_angle": t.o_angle, "o_dihedral": t.o_dihedral,
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
        # Distances are the same for a mirror image, so the signed angles are checked too.
        matched = conv.require_internal_match(entries, rebuilt)
        if len(atoms) <= DISTANCE_CHECK_LIMIT:
            deviation = conv.require_distance_match([a.xyz for a in atoms], rebuilt)
            check = {"kind": "distances", "pairs": len(atoms) * (len(atoms) - 1) // 2,
                     "max_deviation": deviation}
        else:
            bond, angle, dihedral = matched
            check = {"kind": "internal", "bond": bond, "angle": angle, "dihedral": dihedral}
        torsions = residue_torsions(atoms)
    except ValueError as exc:
        raise bad_request(exc) from None
    downloads = []
    if torsions:
        downloads.append({"label": "Download residue table (.csv)", "filename": "residues.csv",
                          "text": io.residue_table_csv(torsions)})
    downloads.append({"label": "Download Z-matrix (.zmat)", "filename": "zmatrix.zmat",
                      "text": io.zmatrix_text(entries, credit("Z-matrix generated"))})
    return {
        "atoms": [atom_json(a) for a in atoms],
        "zmatrix": [zentry_json(e) for e in entries],
        "torsions": torsions_json(torsions),
        "bonds": bonds_json(entries),
        "check": check,
        "downloads": downloads,
    }


@app.post("/api/internal-to-cartesian")
def internal_to_cartesian(req: ZMatrixRequest):
    try:
        entries = io.parse_zmatrix_text(req.text)
        if len(entries) > MAX_ATOMS:
            raise ValueError("the web version handles up to {} atoms".format(MAX_ATOMS))
        coords = conv.zmatrix_to_cartesian(entries)
        conv.require_internal_match(entries, coords)
        atoms = io.assign_residues([conv.Atom(e.name, xyz) for e, xyz in zip(entries, coords)])
        torsions = residue_torsions(atoms)
    except ValueError as exc:
        raise bad_request(exc) from None

    return {
        "atoms": [atom_json(a) for a in atoms],
        "zmatrix": [zentry_json(e) for e in entries],
        "torsions": torsions_json(torsions),
        "bonds": bonds_json(entries),
        "check": internal_check(entries, coords),
        "downloads": [{"label": "Download PDB file", "filename": "structure.pdb",
                       "text": io.pdb_text(atoms, credit("Built from a Z-matrix"))}],
    }


@app.post("/api/backbone-to-cartesian")
def backbone_to_cartesian(req: BackboneRequest):
    try:
        if req.preset:
            if req.preset not in conv.SECONDARY_STRUCTURE_PRESETS:
                raise ValueError("unknown conformation '{}'".format(req.preset))
            if req.count is None:
                raise ValueError("enter the number of residues")
            phi, psi, omega = conv.SECONDARY_STRUCTURE_PRESETS[req.preset]
            residues = [conv.ResidueTorsions(name, i + 1, "A", phi, psi, omega)
                        for i, name in enumerate(io.sequence_to_residues(req.sequence, req.count))]
        else:
            given = req.residues or []
            if not given:
                raise ValueError("enter phi, psi and omega for at least one residue")
            names = io.sequence_to_residues(req.sequence, len(given))
            residues = [
                conv.ResidueTorsions(
                    (r.res_name or names[i]).strip().upper()[:3] or names[i],
                    r.res_seq if r.res_seq is not None else i + 1,
                    (r.chain or "A").strip()[:1] or "A",
                    r.phi, r.psi, r.omega, r.o_bond, r.o_angle, r.o_dihedral,
                    (r.icode or "").strip()[:1].upper())
                for i, r in enumerate(given)
            ]
        residues = conv.complete_residues(residues)
        entries = conv.build_backbone_zmatrix(residues)
        atoms = conv.build_backbone(residues)
        torsion_error = conv.require_torsion_match(atoms, residues)
    except ValueError as exc:
        raise bad_request(exc) from None

    return {
        "atoms": [atom_json(a) for a in atoms],
        "zmatrix": [zentry_json(e) for e in entries],
        "torsions": torsions_json(conv.backbone_torsions(atoms)),
        "input": torsions_json(residues),
        "bonds": bonds_json(entries),
        "check": {"kind": "torsions", "max_error": torsion_error},
        "downloads": [
            {"label": "Download PDB file", "filename": "backbone.pdb",
             "text": io.pdb_text(atoms, credit("Backbone built"))},
            {"label": "Download residue table (.csv)", "filename": "residues.csv",
             "text": io.residue_table_csv(residues)},
        ],
    }


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))


# On Vercel the public/ folder is served by the CDN; locally FastAPI serves it.
if not os.environ.get("VERCEL"):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR), name="public")
