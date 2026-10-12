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
    res_name: str
    res_seq: int
    chain: str
    phi: Optional[float]
    psi: Optional[float]
    omega: Optional[float]


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

PEPTIDE_BOND_MAX = 2.0

SECONDARY_STRUCTURE_PRESETS = {
    "alpha-helix": (-57.0, -47.0, 180.0),
    "beta-strand": (-139.0, 135.0, 180.0),
    "polyproline-II": (-75.0, 145.0, 180.0),
    "extended": (180.0, 180.0, 180.0),
}


def backbone_torsions(atoms: Sequence[Atom]) -> List[ResidueTorsions]:
    """Compute phi, psi and omega for every residue with complete N, CA, C atoms.

    phi(i)   = C(i-1) - N(i) - CA(i) - C(i)
    psi(i)   = N(i) - CA(i) - C(i) - N(i+1)
    omega(i) = CA(i) - C(i) - N(i+1) - CA(i+1)
    Torsions that cross a chain break are reported as None.
    """
    residues = []
    index = {}
    for atom in atoms:
        key = (atom.chain, atom.res_seq)
        if key not in index:
            index[key] = len(residues)
            residues.append((atom.chain, atom.res_seq, atom.res_name, {}))
        if atom.name in BACKBONE_ATOMS:
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
        result.append(ResidueTorsions(res_name, seq, chain, phi, psi, omega))
    return result


def build_backbone_zmatrix(res_names: Sequence[str], phis: Sequence[float],
                           psis: Sequence[float], omegas: Sequence[float]) -> List[ZEntry]:
    """Z-matrix for an N-CA-C backbone with ideal bond lengths and angles.

    phis[0] and the last residue's psi and omega are not needed to place any
    atom, so their values are ignored.
    """
    entries: List[ZEntry] = []
    for i in range(len(res_names)):
        n_idx = 3 * i
        if i == 0:
            entries.append(ZEntry("N"))
            entries.append(ZEntry("CA", 0, BOND_N_CA))
            entries.append(ZEntry("C", 1, BOND_CA_C, 0, ANGLE_N_CA_C))
            continue
        prev_n, prev_ca, prev_c = n_idx - 3, n_idx - 2, n_idx - 1
        entries.append(ZEntry("N", prev_c, BOND_C_N, prev_ca, ANGLE_CA_C_N, prev_n, psis[i - 1]))
        entries.append(ZEntry("CA", n_idx, BOND_N_CA, prev_c, ANGLE_C_N_CA, prev_ca, omegas[i - 1]))
        entries.append(ZEntry("C", n_idx + 1, BOND_CA_C, n_idx, ANGLE_N_CA_C, prev_c, phis[i]))
    return entries


def build_backbone(res_names: Sequence[str], phis: Sequence[float],
                   psis: Sequence[float], omegas: Sequence[float]) -> List[Atom]:
    """Cartesian N, CA, C atoms for a peptide with the given backbone torsions."""
    entries = build_backbone_zmatrix(res_names, phis, psis, omegas)
    coords = zmatrix_to_cartesian(entries)
    return [
        Atom(entry.name, xyz, res_names[i // 3], i // 3 + 1)
        for i, (entry, xyz) in enumerate(zip(entries, coords))
    ]


def backbone_torsion_error(atoms: Sequence[Atom], phis: Sequence[Optional[float]],
                           psis: Sequence[Optional[float]], omegas: Sequence[Optional[float]]) -> float:
    """Largest difference between requested torsions and those measured from atoms."""
    worst = 0.0
    for t, phi, psi, omega in zip(backbone_torsions(atoms), phis, psis, omegas):
        for got, want in ((t.phi, phi), (t.psi, psi), (t.omega, omega)):
            if got is not None and want is not None:
                worst = max(worst, abs(geo.angle_difference(got, want)))
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
