"""Cartesian <-> internal coordinate conversions, plus protein backbone helpers.

Internal coordinates are stored as a Z-matrix: every atom after the first is
positioned by a bond length to one earlier atom, a bond angle with a second
earlier atom and a dihedral angle with a third earlier atom.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from . import geometry as geo

Vec3 = Tuple[float, float, float]


@dataclass
class Atom:
    name: str
    xyz: Vec3
    res_name: str = "UNK"
    res_seq: int = 1
    chain: str = "A"


@dataclass
class ZEntry:
    """One Z-matrix row. Reference indices are 0-based."""

    name: str
    bond_ref: Optional[int] = None
    bond: Optional[float] = None
    angle_ref: Optional[int] = None
    angle: Optional[float] = None
    dihedral_ref: Optional[int] = None
    dihedral: Optional[float] = None


@dataclass
class ResidueTorsions:
    """Internal coordinates of one residue: backbone torsions plus its carbonyl oxygen.

    o_bond is the C=O length, o_angle the CA-C-O angle and o_dihedral the
    N-CA-C-O dihedral.
    """

    res_name: str
    res_seq: int
    chain: str
    phi: Optional[float]
    psi: Optional[float]
    omega: Optional[float]
    o_bond: Optional[float] = None
    o_angle: Optional[float] = None
    o_dihedral: Optional[float] = None


# ---------------------------------------------------------------------------
# Cartesian -> internal
# ---------------------------------------------------------------------------

# Bonded partners in a PDB file sit close together in the atom list, so only
# this many preceding atoms are searched when choosing references.
SEARCH_WINDOW = 100
COLLINEAR_TOLERANCE = 5.0


def _collinear(a: Vec3, b: Vec3, c: Vec3) -> bool:
    try:
        angle = geo.bond_angle(a, b, c)
    except ValueError:
        return True
    return angle < COLLINEAR_TOLERANCE or angle > 180.0 - COLLINEAR_TOLERANCE


def _pick_reference(coords, near, before, exclude=(), line=None) -> int:
    """Earlier atom closest to coords[near], preferring one not collinear with `line`."""
    window = range(max(0, before - SEARCH_WINDOW), before)
    candidates = [m for m in window if m not in exclude] or [m for m in range(before) if m not in exclude]
    candidates.sort(key=lambda m: geo.distance(coords[m], coords[near]))
    if line is not None:
        for m in candidates:
            if not _collinear(coords[line[0]], coords[line[1]], coords[m]):
                return m
    return candidates[0]


def cartesian_to_zmatrix(atoms: Sequence[Atom]) -> List[ZEntry]:
    """Build a Z-matrix whose references follow the bonding network.

    Each atom is bonded to its closest earlier atom j; the angle reference is
    the atom that j was bonded to, and the dihedral reference is the atom that
    one was bonded to. For an N-CA-C protein backbone the dihedrals are then
    exactly psi, omega and phi.
    """
    coords = [atom.xyz for atom in atoms]
    entries: List[ZEntry] = []
    for i, atom in enumerate(atoms):
        entry = ZEntry(atom.name)
        try:
            if i >= 1:
                j = _pick_reference(coords, near=i, before=i)
                entry.bond_ref, entry.bond = j, geo.distance(coords[i], coords[j])
            if i >= 2:
                k = entries[j].bond_ref
                if k is None or _collinear(coords[i], coords[j], coords[k]):
                    k = _pick_reference(coords, near=j, before=i, exclude={j}, line=(i, j))
                entry.angle_ref, entry.angle = k, geo.bond_angle(coords[i], coords[j], coords[k])
            if i >= 3:
                l = entries[k].bond_ref
                if l is None or l == j or _collinear(coords[j], coords[k], coords[l]):
                    l = _pick_reference(coords, near=k, before=i, exclude={j, k}, line=(j, k))
                entry.dihedral_ref = l
                entry.dihedral = geo.dihedral(coords[i], coords[j], coords[k], coords[l])
        except ValueError as exc:
            raise ValueError("atom {} ({}): {}".format(i + 1, atom.name, exc)) from None
        entries.append(entry)
    return entries


# ---------------------------------------------------------------------------
# Internal -> Cartesian
# ---------------------------------------------------------------------------

def validate_zmatrix(entries: Sequence[ZEntry]) -> None:
    """Raise ValueError describing the first problem found in the Z-matrix."""
    for i, e in enumerate(entries):
        label = "atom {} ({})".format(i + 1, e.name)
        needed = [("bond", e.bond_ref, e.bond), ("angle", e.angle_ref, e.angle),
                  ("dihedral", e.dihedral_ref, e.dihedral)][: min(i, 3)]
        refs = []
        for kind, ref, value in needed:
            if ref is None or value is None:
                raise ValueError("{} is missing its {} value".format(label, kind))
            if not 0 <= ref < i:
                raise ValueError("{}: {} reference must be an earlier atom (1-{})".format(label, kind, i))
            refs.append(ref)
        if len(set(refs)) != len(refs):
            raise ValueError("{}: bond, angle and dihedral references must be different atoms".format(label))
        if i >= 1 and e.bond <= 0:
            raise ValueError("{}: bond length must be positive".format(label))
        if i >= 2 and not 0 < e.angle <= 180:
            raise ValueError("{}: bond angle must be between 0 and 180 degrees".format(label))


def zmatrix_to_cartesian(entries: Sequence[ZEntry]) -> List[Vec3]:
    """Place atoms from a Z-matrix.

    Atom 1 sits at the origin, atom 2 on the +x axis and atom 3 in the xy-plane;
    every later atom is placed with the NeRF algorithm.
    """
    validate_zmatrix(entries)
    coords: List[Vec3] = []
    for i, e in enumerate(entries):
        try:
            if i == 0:
                xyz = (0.0, 0.0, 0.0)
            elif i == 1:
                xyz = geo.add(coords[e.bond_ref], (e.bond, 0.0, 0.0))
            elif i == 2:
                j, k = coords[e.bond_ref], coords[e.angle_ref]
                # Atoms 1 and 2 lie on the x-axis, so this helper point fixes the xy-plane.
                helper = geo.add(k, (0.0, 1.0, 0.0))
                xyz = geo.place_atom(helper, k, j, e.bond, e.angle, 0.0)
            else:
                xyz = geo.place_atom(
                    coords[e.dihedral_ref], coords[e.angle_ref], coords[e.bond_ref],
                    e.bond, e.angle, e.dihedral,
                )
        except ValueError as exc:
            raise ValueError("atom {} ({}): {}".format(i + 1, e.name, exc)) from None
        coords.append(xyz)
    return coords


# ---------------------------------------------------------------------------
# Round-trip checks
# ---------------------------------------------------------------------------

def zmatrix_errors(entries: Sequence[ZEntry], coords: Sequence[Vec3]) -> Tuple[float, float, float]:
    """Largest bond, angle and dihedral error between a Z-matrix and coordinates."""
    max_bond = max_angle = max_dihedral = 0.0
    for i, e in enumerate(entries):
        if i >= 1:
            d = geo.distance(coords[i], coords[e.bond_ref])
            max_bond = max(max_bond, abs(d - e.bond))
        if i >= 2:
            a = geo.bond_angle(coords[i], coords[e.bond_ref], coords[e.angle_ref])
            max_angle = max(max_angle, abs(a - e.angle))
        if i >= 3:
            t = geo.dihedral(coords[i], coords[e.bond_ref], coords[e.angle_ref], coords[e.dihedral_ref])
            max_dihedral = max(max_dihedral, abs(geo.angle_difference(t, e.dihedral)))
    return max_bond, max_angle, max_dihedral


def max_distance_deviation(a: Sequence[Vec3], b: Sequence[Vec3]) -> float:
    """Largest difference in any interatomic distance between two structures.

    Zero means the structures are identical up to a rigid-body rotation and
    translation.
    """
    worst = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            worst = max(worst, abs(geo.distance(a[i], a[j]) - geo.distance(b[i], b[j])))
    return worst


# ---------------------------------------------------------------------------
# Protein backbone
# ---------------------------------------------------------------------------

BACKBONE_ATOMS = ("N", "CA", "C")

# Ideal backbone geometry from Engh & Huber (1991).
BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_N = 1.329
ANGLE_N_CA_C = 111.2
ANGLE_CA_C_N = 116.2
ANGLE_C_N_CA = 121.7
BOND_C_O = 1.231
ANGLE_CA_C_O = 120.5

PEPTIDE_BOND_MAX = 2.0

SECONDARY_STRUCTURE_PRESETS = {
    "alpha-helix": (-57.0, -47.0, 180.0),
    "beta-strand": (-139.0, 135.0, 180.0),
    "polyproline-II": (-75.0, 145.0, 180.0),
    "extended": (180.0, 180.0, 180.0),
}


def backbone_torsions(atoms: Sequence[Atom]) -> List[ResidueTorsions]:
    """Internal coordinates of every residue with complete N, CA, C atoms.

    phi(i)   = C(i-1) - N(i) - CA(i) - C(i)
    psi(i)   = N(i) - CA(i) - C(i) - N(i+1)
    omega(i) = CA(i) - C(i) - N(i+1) - CA(i+1)
    If the residue has its carbonyl O, the C=O length, CA-C-O angle and
    N-CA-C-O dihedral are included. Torsions that cross a chain break are None.
    """
    residues = []
    index = {}
    for atom in atoms:
        key = (atom.chain, atom.res_seq)
        if key not in index:
            index[key] = len(residues)
            residues.append((atom.chain, atom.res_seq, atom.res_name, {}))
        if atom.name in BACKBONE_ATOMS or atom.name == "O":
            residues[index[key]][3].setdefault(atom.name, atom.xyz)

    complete = [r for r in residues if all(name in r[3] for name in BACKBONE_ATOMS)]

    def linked(r1, r2):
        return r1[0] == r2[0] and geo.distance(r1[3]["C"], r2[3]["N"]) < PEPTIDE_BOND_MAX

    result = []
    for i, (chain, seq, res_name, bb) in enumerate(complete):
        prev = complete[i - 1] if i > 0 and linked(complete[i - 1], complete[i]) else None
        nxt = complete[i + 1] if i + 1 < len(complete) and linked(complete[i], complete[i + 1]) else None
        phi = geo.dihedral(prev[3]["C"], bb["N"], bb["CA"], bb["C"]) if prev else None
        psi = geo.dihedral(bb["N"], bb["CA"], bb["C"], nxt[3]["N"]) if nxt else None
        omega = geo.dihedral(bb["CA"], bb["C"], nxt[3]["N"], nxt[3]["CA"]) if nxt else None
        torsions = ResidueTorsions(res_name, seq, chain, phi, psi, omega)
        if "O" in bb:
            torsions.o_bond = geo.distance(bb["C"], bb["O"])
            torsions.o_angle = geo.bond_angle(bb["CA"], bb["C"], bb["O"])
            torsions.o_dihedral = geo.dihedral(bb["N"], bb["CA"], bb["C"], bb["O"])
        result.append(torsions)
    return result


def oxygen_dihedral_from_psi(psi: float) -> float:
    """N-CA-C-O that keeps O in the peptide plane, opposite N(i+1)."""
    value = geo.angle_difference(psi + 180.0, 0.0)
    return 180.0 if value == -180.0 else value


def complete_residues(residues: Sequence[ResidueTorsions]) -> List[ResidueTorsions]:
    """Check that every residue has the angles needed to build it and fill in oxygen defaults.

    Residue 1 needs no phi and the last residue needs no psi or omega, because
    those angles involve atoms of a neighbouring residue that does not exist.
    The oxygen defaults are C=O 1.231 A, CA-C-O 120.5 deg and N-CA-C-O = psi + 180.
    """
    if not residues:
        raise ValueError("enter at least one residue")
    count = len(residues)
    completed = []
    for i, r in enumerate(residues):
        label = "residue {} ({}{})".format(i + 1, r.res_name, r.res_seq)
        needed = ([("phi", r.phi)] if i > 0 else []) + (
            [("psi", r.psi), ("omega", r.omega)] if i < count - 1 else [])
        missing = [name for name, value in needed if value is None]
        if missing:
            raise ValueError("{} is missing {}".format(label, " and ".join(missing)))
        o_dihedral = r.o_dihedral
        if o_dihedral is None:
            if r.psi is None:
                raise ValueError("{} needs psi or an N-CA-C-O dihedral to place its oxygen".format(label))
            o_dihedral = oxygen_dihedral_from_psi(r.psi)
        o_bond = BOND_C_O if r.o_bond is None else r.o_bond
        o_angle = ANGLE_CA_C_O if r.o_angle is None else r.o_angle
        if o_bond <= 0:
            raise ValueError("{}: the C=O bond length must be positive".format(label))
        if not 0 < o_angle < 180:
            raise ValueError("{}: the CA-C-O angle must be between 0 and 180 degrees".format(label))
        completed.append(ResidueTorsions(
            r.res_name, r.res_seq, r.chain,
            r.phi if i > 0 else None,
            r.psi,
            r.omega if i < count - 1 else None,
            o_bond, o_angle, o_dihedral,
        ))
    return completed


def build_backbone_zmatrix(residues: Sequence[ResidueTorsions]) -> List[ZEntry]:
    """Z-matrix for an N, CA, C, O backbone with ideal N-CA-C geometry.

    Atoms are ordered N, CA, C, O within each residue, as in a PDB file.
    """
    residues = complete_residues(residues)
    entries: List[ZEntry] = []
    for i, r in enumerate(residues):
        n, ca, c = 4 * i, 4 * i + 1, 4 * i + 2
        if i == 0:
            entries.append(ZEntry("N"))
            entries.append(ZEntry("CA", n, BOND_N_CA))
            entries.append(ZEntry("C", ca, BOND_CA_C, n, ANGLE_N_CA_C))
        else:
            prev = residues[i - 1]
            prev_n, prev_ca, prev_c = n - 4, n - 3, n - 2
            entries.append(ZEntry("N", prev_c, BOND_C_N, prev_ca, ANGLE_CA_C_N, prev_n, prev.psi))
            entries.append(ZEntry("CA", n, BOND_N_CA, prev_c, ANGLE_C_N_CA, prev_ca, prev.omega))
            entries.append(ZEntry("C", ca, BOND_CA_C, n, ANGLE_N_CA_C, prev_c, r.phi))
        entries.append(ZEntry("O", c, r.o_bond, ca, r.o_angle, n, r.o_dihedral))
    return entries


def build_backbone(residues: Sequence[ResidueTorsions]) -> List[Atom]:
    """Cartesian N, CA, C, O atoms for a peptide with the given internal coordinates."""
    entries = build_backbone_zmatrix(residues)
    coords = zmatrix_to_cartesian(entries)
    return [
        Atom(entry.name, xyz, residues[i // 4].res_name, residues[i // 4].res_seq, residues[i // 4].chain)
        for i, (entry, xyz) in enumerate(zip(entries, coords))
    ]


def backbone_torsion_error(atoms: Sequence[Atom], residues: Sequence[ResidueTorsions]) -> float:
    """Largest difference (degrees) between requested angles and those measured from atoms."""
    worst = 0.0
    for got, want in zip(backbone_torsions(atoms), complete_residues(residues)):
        for measured, requested in ((got.phi, want.phi), (got.psi, want.psi), (got.omega, want.omega),
                                    (got.o_angle, want.o_angle), (got.o_dihedral, want.o_dihedral)):
            if measured is not None and requested is not None:
                worst = max(worst, abs(geo.angle_difference(measured, requested)))
    return worst


def ramachandran_region(phi: Optional[float], psi: Optional[float]) -> str:
    """Rough secondary-structure region of a (phi, psi) pair."""
    if phi is None or psi is None:
        return "-"
    if -180 <= phi <= -45 and (psi >= 90 or psi <= -150):
        return "beta"
    if -160 <= phi <= -20 and -120 <= psi <= 50:
        return "alpha-R"
    if 20 <= phi <= 120 and -20 <= psi <= 100:
        return "alpha-L"
    return "other"
