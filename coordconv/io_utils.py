"""Reading and writing coordinates: typed-in lines, PDB files, Z-matrix files and residue tables."""

import csv
import io as _io
import os
import re
from typing import List, Optional, Sequence, Tuple

from .conversions import Atom, ResidueTorsions, ZEntry, Vec3

BACKBONE_WITH_OXYGEN = ("N", "CA", "C", "O")

ONE_TO_THREE = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}


def clean_path(raw: str) -> str:
    """Normalise a path typed or drag-and-dropped into the terminal."""
    path = raw.strip().strip("'\"").replace("\\ ", " ")
    return os.path.abspath(os.path.expanduser(path))


def _float(token: str, what: str) -> float:
    try:
        return float(token)
    except ValueError:
        raise ValueError("'{}' is not a valid number for {}".format(token, what)) from None


def _tokens(line: str) -> List[str]:
    return line.replace(",", " ").split()


# ---------------------------------------------------------------------------
# Typed-in input
# ---------------------------------------------------------------------------

def parse_cartesian_line(line: str, default_name: str) -> Tuple[str, Vec3]:
    """Parse 'NAME X Y Z' (or just 'X Y Z')."""
    tokens = _tokens(line)
    if len(tokens) == 3:
        name, nums = default_name, tokens
    elif len(tokens) == 4:
        name, nums = tokens[0], tokens[1:]
    else:
        raise ValueError("expected NAME X Y Z (e.g. 'CA 1.458 0.000 0.000')")
    x, y, z = (_float(t, axis) for t, axis in zip(nums, "xyz"))
    return name, (x, y, z)


def zmatrix_hint(i: int) -> str:
    return ["NAME", "NAME  bond", "NAME  bond  angle", "NAME  bond  angle  dihedral"][min(i, 3)]


def parse_zmatrix_line(line: str, i: int) -> ZEntry:
    """Parse the Z-matrix row for atom index i (0-based).

    Two layouts are accepted:
      short:  NAME bond angle dihedral                 (references = previous atoms)
      full:   NAME ref bond ref angle ref dihedral     (references are 1-based atom numbers)
    """
    tokens = _tokens(line)
    if not tokens:
        raise ValueError("empty line")
    name, nums = tokens[0], tokens[1:]
    n_values = min(i, 3)

    if len(nums) == n_values:
        refs = [i - 1, i - 2, i - 3][:n_values]
        values = [_float(t, k) for t, k in zip(nums, ("bond", "angle", "dihedral"))]
    elif len(nums) == 2 * n_values and n_values > 0:
        refs, values = [], []
        for k, kind in enumerate(("bond", "angle", "dihedral")[:n_values]):
            ref_token, value_token = nums[2 * k], nums[2 * k + 1]
            try:
                refs.append(int(ref_token) - 1)
            except ValueError:
                raise ValueError("'{}' is not a valid atom number for the {} reference".format(ref_token, kind)) from None
            values.append(_float(value_token, kind))
    else:
        raise ValueError("atom {} needs: {}".format(i + 1, zmatrix_hint(i)))

    refs += [None] * (3 - len(refs))
    values += [None] * (3 - len(values))
    return ZEntry(name, refs[0], values[0], refs[1], values[1], refs[2], values[2])


def assign_residues(atoms: List[Atom]) -> List[Atom]:
    """Number residues for atoms without residue info: a new residue starts at each N."""
    res_seq = 0
    for i, atom in enumerate(atoms):
        if atom.name.upper() == "N" or i == 0:
            res_seq += 1
        atom.res_seq = res_seq
    return atoms


def sequence_to_residues(seq: str, count: int) -> List[str]:
    """Three-letter names from a one-letter sequence, padded with ALA."""
    seq = "".join(seq.split()).upper()
    unknown = sorted({c for c in seq if c not in ONE_TO_THREE})
    if unknown:
        raise ValueError("unknown one-letter code(s): {}".format(", ".join(unknown)))
    names = [ONE_TO_THREE[c] for c in seq[:count]]
    return names + ["ALA"] * (count - len(names))


# ---------------------------------------------------------------------------
# PDB files
# ---------------------------------------------------------------------------

def looks_like_pdb(text: str) -> bool:
    return any(line.startswith(("ATOM ", "HETATM")) for line in text.splitlines())


def parse_pdb_text(text: str, backbone_only: bool = False) -> List[Atom]:
    """Read ATOM records from the first model of PDB-formatted text."""
    atoms = []
    for line in text.splitlines():
        record = line[:6].strip()
        if record == "ENDMDL" and atoms:
            break
        if record != "ATOM":
            continue
        if line[16:17] not in (" ", "A", ""):
            continue
        name = line[12:16].strip()
        if backbone_only and name not in BACKBONE_WITH_OXYGEN:
            continue
        try:
            atoms.append(Atom(
                name=name,
                xyz=(float(line[30:38]), float(line[38:46]), float(line[46:54])),
                res_name=line[17:20].strip() or "UNK",
                res_seq=int(line[22:26]),
                chain=line[21:22].strip() or "A",
            ))
        except ValueError:
            raise ValueError("could not read this PDB line: {}".format(line.rstrip())) from None
    if not atoms:
        raise ValueError("no ATOM records found")
    return atoms


def read_pdb(path: str, backbone_only: bool = False) -> List[Atom]:
    with open(path) as fh:
        return parse_pdb_text(fh.read(), backbone_only)


def parse_cartesian_text(text: str, backbone_only: bool = False) -> List[Atom]:
    """Atoms from either PDB text or 'NAME X Y Z' lines (# starts a comment)."""
    if looks_like_pdb(text):
        return parse_pdb_text(text, backbone_only)
    atoms = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        try:
            name, xyz = parse_cartesian_line(content, "A{}".format(len(atoms) + 1))
        except ValueError as exc:
            raise ValueError("line {}: {}".format(line_no, exc)) from None
        atoms.append(Atom(name, xyz))
    if backbone_only:
        atoms = [a for a in atoms if a.name in BACKBONE_WITH_OXYGEN]
    return assign_residues(atoms)


def _pdb_atom_name(name: str) -> str:
    return name[:4] if len(name) >= 4 else " " + name.ljust(3)


def _element(name: str) -> str:
    letters = "".join(c for c in name if c.isalpha())
    return letters[:1].upper() if letters else "X"


def pdb_text(atoms: Sequence[Atom], title: Optional[str] = None) -> str:
    lines = []
    if title:
        lines.append("REMARK   1 {}".format(title[:68]))
    for serial, atom in enumerate(atoms, start=1):
        x, y, z = atom.xyz
        lines.append(
            "ATOM  {:5d} {:4s} {:>3s} {:1s}{:4d}    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}".format(
                serial, _pdb_atom_name(atom.name), atom.res_name[:3], atom.chain[:1],
                atom.res_seq, x, y, z, 1.0, 0.0, _element(atom.name),
            )
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pdb(atoms: Sequence[Atom], path: str, title: Optional[str] = None) -> None:
    with open(path, "w") as fh:
        fh.write(pdb_text(atoms, title))


# ---------------------------------------------------------------------------
# Z-matrix files
# ---------------------------------------------------------------------------

def format_zmatrix_line(entry: ZEntry) -> str:
    parts = ["{:<5s}".format(entry.name)]
    for ref, value, fmt in ((entry.bond_ref, entry.bond, "{:9.4f}"),
                            (entry.angle_ref, entry.angle, "{:10.3f}"),
                            (entry.dihedral_ref, entry.dihedral, "{:10.3f}")):
        if ref is None:
            break
        parts.append("{:4d} ".format(ref + 1) + fmt.format(value))
    return "  ".join(parts).rstrip()


def parse_zmatrix_text(text: str) -> List[ZEntry]:
    entries = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        try:
            entries.append(parse_zmatrix_line(content, len(entries)))
        except ValueError as exc:
            raise ValueError("line {}: {}".format(line_no, exc)) from None
    if not entries:
        raise ValueError("no Z-matrix rows found")
    return entries


def read_zmatrix(path: str) -> List[ZEntry]:
    with open(path) as fh:
        return parse_zmatrix_text(fh.read())


def zmatrix_text(entries: Sequence[ZEntry], title: Optional[str] = None) -> str:
    lines = ["# {}".format(title)] if title else []
    lines.append("# NAME  bond_ref bond(A)  angle_ref angle(deg)  dihedral_ref dihedral(deg)")
    lines += [format_zmatrix_line(entry) for entry in entries]
    return "\n".join(lines) + "\n"


def write_zmatrix(entries: Sequence[ZEntry], path: str, title: Optional[str] = None) -> None:
    with open(path, "w") as fh:
        fh.write(zmatrix_text(entries, title))


# ---------------------------------------------------------------------------
# Residue tables (phi, psi, omega and carbonyl oxygen per residue)
# ---------------------------------------------------------------------------

RESIDUE_COLUMNS = ("residue", "phi", "psi", "omega", "c_o_bond", "ca_c_o_angle", "n_ca_c_o_dihedral")


def _cell(value: Optional[float], digits: int) -> str:
    return "" if value is None else "{:.{}f}".format(value, digits)


def residue_table_csv(residues: Sequence[ResidueTorsions]) -> str:
    """CSV with one row per residue; opens in Excel and loads back into the builder."""
    buffer = _io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(RESIDUE_COLUMNS)
    for r in residues:
        writer.writerow([
            "{}{}".format(r.res_name, r.res_seq),
            _cell(r.phi, 2), _cell(r.psi, 2), _cell(r.omega, 2),
            _cell(r.o_bond, 3), _cell(r.o_angle, 2), _cell(r.o_dihedral, 2),
        ])
    return buffer.getvalue()


def parse_residue_csv(text: str) -> List[ResidueTorsions]:
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    fields = [f.strip().lower() for f in (reader.fieldnames or [])]
    missing = [c for c in ("residue", "phi", "psi", "omega") if c not in fields]
    if missing:
        raise ValueError("the residue table needs the columns: {}".format(", ".join(RESIDUE_COLUMNS)))

    def number(row, key, row_no):
        raw = (row.get(key) or "").strip()
        if raw in ("", "-"):
            return None
        return _float(raw, "{} in row {}".format(key, row_no))

    residues = []
    for row_no, raw_row in enumerate(reader, start=1):
        row = {k.strip().lower(): v for k, v in raw_row.items() if k}
        match = re.match(r"^\s*([A-Za-z]+)\s*(-?\d+)?\s*$", row.get("residue") or "")
        if not match:
            raise ValueError("row {}: residue should look like ALA1".format(row_no))
        residues.append(ResidueTorsions(
            res_name=match.group(1).upper()[:3],
            res_seq=int(match.group(2)) if match.group(2) else row_no,
            chain="A",
            phi=number(row, "phi", row_no),
            psi=number(row, "psi", row_no),
            omega=number(row, "omega", row_no),
            o_bond=number(row, "c_o_bond", row_no),
            o_angle=number(row, "ca_c_o_angle", row_no),
            o_dihedral=number(row, "n_ca_c_o_dihedral", row_no),
        ))
    if not residues:
        raise ValueError("the residue table has no rows")
    return residues


def read_residue_csv(path: str) -> List[ResidueTorsions]:
    with open(path) as fh:
        return parse_residue_csv(fh.read())
