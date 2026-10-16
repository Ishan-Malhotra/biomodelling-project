#!/usr/bin/env python3
"""Biomolecular coordinate converter.

Interactive terminal that converts protein structures between Cartesian
coordinates (x, y, z) and internal coordinates (bond lengths, bond angles and
dihedral angles).

Run with:  python3 main.py
"""

import os
import sys
from datetime import datetime

from coordconv import conversions as conv
from coordconv import io_utils as io
from settings import STRUCTURE_TYPE, STUDENT_ID, STUDENT_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")

WIDTH = 66
PREVIEW_ROWS = 20
# The all-pairs distance check is O(n^2), so larger structures use a cheaper check.
DISTANCE_CHECK_LIMIT = 1500

USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ


class QuitProgram(Exception):
    pass


# ---------------------------------------------------------------------------
# Terminal helpers
# ---------------------------------------------------------------------------

def style(text, *codes):
    if not USE_COLOR:
        return text
    return "\033[{}m{}\033[0m".format(";".join(codes), text)


def bold(text):
    return style(text, "1")


def accent(text):
    return style(text, "1", "36")


def good(text):
    return style(text, "32")


def warn(text):
    return style(text, "33")


def bad(text):
    return style(text, "31")


def muted(text):
    return style(text, "2")


def clear():
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="")
    else:
        print()


def ask(prompt):
    try:
        answer = input(accent("  > ") + prompt + " ")
    except EOFError:
        raise QuitProgram from None
    if answer.strip().lower() in ("q", "quit", "exit"):
        raise QuitProgram
    return answer.strip()


def pause():
    ask(muted("Press Enter to continue..."))


def banner(subtitle=None):
    clear()
    inner = WIDTH - 2

    def row(text="", painter=str, center=False):
        padded = text.center(inner - 2) if center else text.ljust(inner - 2)
        print(accent("║") + " " + painter(padded) + " " + accent("║"))

    print(accent("╔" + "═" * inner + "╗"))
    row("BIOMOLECULAR COORDINATE CONVERTER", bold, center=True)
    row("Cartesian (x, y, z)  <->  Internal (φ, ψ, ω + oxygen)", muted, center=True)
    print(accent("╠" + "═" * inner + "╣"))
    row("Name      : " + STUDENT_NAME)
    row("ID number : " + STUDENT_ID)
    row("Structure : " + STRUCTURE_TYPE)
    print(accent("╚" + "═" * inner + "╝"))
    if subtitle:
        print()
        print("  " + bold(subtitle))
        print("  " + muted("─" * len(subtitle)))
    print()


def menu(options):
    for option in options:
        key, label = option[0], option[1]
        print("  " + accent("[{}]".format(key)) + " " + label)
        if len(option) > 2:
            print("      " + muted(option[2]))
    print()


def choose(options, prompt="Choose an option:"):
    valid = {option[0].lower() for option in options}
    while True:
        answer = ask(prompt).lower()
        if answer in valid:
            return answer
        print(bad("    Please enter one of: " + ", ".join(option[0] for option in options)))


def ask_int(prompt, low, high):
    while True:
        raw = ask(prompt)
        if raw.lower() in ("b", "back", "cancel"):
            return None
        try:
            value = int(raw)
        except ValueError:
            value = None
        if value is not None and low <= value <= high:
            return value
        print(bad("    Please enter a whole number from {} to {} (or 'b' to go back).".format(low, high)))


def resolve_input_path(raw):
    path = io.clean_path(raw)
    if not os.path.exists(path):
        alternative = os.path.join(HERE, raw.strip().strip("'\""))
        if os.path.exists(alternative):
            return alternative
    return path


def save_prompt(default_name, writer):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    default_path = os.path.join(OUTPUT_DIR, default_name)
    raw = ask("Save as (Enter = {}):".format(os.path.relpath(default_path)))
    path = io.clean_path(raw) if raw else default_path
    try:
        writer(path)
    except OSError as exc:
        print(bad("    Could not save: {}".format(exc)))
    else:
        print(good("    Saved to {}".format(path)))
    print()


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def fnum(value, digits):
    return "-" if value is None else "{:.{}f}".format(value, digits)


def print_table(headers, rows, aligns, limit=None, kinds=None):
    shown = rows if limit is None or len(rows) <= limit else rows[:limit]
    widths = [max([len(h)] + [len(r[i]) for r in shown]) for i, h in enumerate(headers)]
    if kinds:
        widths = [max(w, len(kinds[i] or "")) for i, w in enumerate(widths)]

    def fmt(cells):
        return "  " + "  ".join(
            c.rjust(w) if a == "r" else c.ljust(w) for c, w, a in zip(cells, widths, aligns)
        )

    if kinds:
        print(muted(fmt([(k or "") for k in kinds])))
    print(bold(fmt(headers)))
    print(muted("  " + "  ".join("─" * w for w in widths)))
    for r in shown:
        print(fmt(r))
    if len(shown) < len(rows):
        print(muted("  ... and {} more rows".format(len(rows) - len(shown))))
    print()


def print_cartesian_table(atoms, limit=None):
    rows = [
        [str(i), a.name, io.residue_id(a.res_name, a.res_seq, a.icode), fnum(a.xyz[0], 3), fnum(a.xyz[1], 3), fnum(a.xyz[2], 3)]
        for i, a in enumerate(atoms, start=1)
    ]
    print_table(["#", "Atom", "Residue", "x (Å)", "y (Å)", "z (Å)"], rows, "rllrrr", limit)


def print_zmatrix_table(entries, atoms=None, limit=None):
    def ref(index):
        return "-" if index is None else "{}:{}".format(index + 1, entries[index].name)

    rows = []
    for i, e in enumerate(entries):
        row = [str(i + 1), e.name]
        if atoms:
            row.append("{}{}".format(atoms[i].res_name, atoms[i].res_seq))
        row += [ref(e.bond_ref), fnum(e.bond, 4), ref(e.angle_ref), fnum(e.angle, 2),
                ref(e.dihedral_ref), fnum(e.dihedral, 2)]
        rows.append(row)
    headers = ["#", "Atom"] + (["Residue"] if atoms else []) + [
        "Bonded to", "Length (Å)", "Angle with", "Angle (°)", "Dihedral with", "Dihedral (°)"]
    aligns = "rl" + ("l" if atoms else "") + "lrlrlr"
    print_table(headers, rows, aligns, limit)


def has_torsions(torsions):
    return any(t.phi is not None or t.psi is not None or t.o_dihedral is not None for t in torsions)


def print_torsion_table(torsions, limit=None):
    """Per-residue internal coordinates: phi, psi, omega and the carbonyl oxygen."""
    with_oxygen = any(t.o_dihedral is not None for t in torsions)
    headers = ["Residue", "phi φ (°)", "psi ψ (°)", "omega ω (°)"]
    kinds = ["", "", "", ""]
    aligns = "lrrr"
    if with_oxygen:
        headers += ["C=O (Å)", "CA-C-O (°)", "N-CA-C-O (°)"]
        kinds += ["bond length", "bond angle", "dihedral (torsion)"]
        aligns += "rrr"
    headers.append("Region")
    kinds.append("")
    aligns += "l"
    rows = []
    for t in torsions:
        row = [io.residue_id(t.res_name, t.res_seq, t.icode), fnum(t.phi, 1), fnum(t.psi, 1), fnum(t.omega, 1)]
        if with_oxygen:
            row += [fnum(t.o_bond, 3), fnum(t.o_angle, 1), fnum(t.o_dihedral, 1)]
        row.append(conv.ramachandran_region(t.phi, t.psi))
        rows.append(row)
    print_table(headers, rows, aligns, limit, kinds=kinds)


# ---------------------------------------------------------------------------
# Cartesian -> internal
# ---------------------------------------------------------------------------

BACKBONE_CYCLE = ("N", "CA", "C", "O")


def next_backbone_atom(atoms):
    for atom in reversed(atoms):
        if atom.name in BACKBONE_CYCLE:
            return BACKBONE_CYCLE[(BACKBONE_CYCLE.index(atom.name) + 1) % len(BACKBONE_CYCLE)]
    return "N"


def type_cartesian():
    banner("Cartesian → Internal: enter coordinates")
    print("  The program asks for the N, CA (alpha carbon), C and O of each residue")
    print("  in turn. Type the x, y and z of each atom in Ångström.\n")
    print(muted("  Example:   Residue 1  N   > 0.000  0.000  0.000"))
    print(muted("             Residue 1  CA  > 1.458  0.000  0.000\n"))
    print(muted("  To enter a different atom, type its name first:  CB 18.170 12.703 5.337"))
    print(muted("  Commands: press Enter (or type 'done') to finish, 'undo' to remove"))
    print(muted("            the last atom, 'cancel' to go back.\n"))

    atoms = []
    while True:
        expected = next_backbone_atom(atoms)
        residue = sum(1 for a in atoms if a.name == "N") + (1 if expected == "N" else 0)
        raw = ask("Residue {}  {:<3s} [x y z]:".format(max(residue, 1), expected))
        command = raw.lower()
        if command in ("", "done"):
            if len(atoms) >= 2:
                return io.assign_residues(atoms)
            print(warn("    Enter at least 2 atoms - internal coordinates start with a bond."))
            continue
        if command == "undo":
            if atoms:
                removed = atoms.pop()
                print(muted("    Removed atom {} ({}).".format(len(atoms) + 1, removed.name)))
            continue
        if command in ("cancel", "b", "back"):
            return None
        try:
            name, xyz = io.parse_cartesian_line(raw, expected)
        except ValueError as exc:
            print(bad("    {}".format(exc)))
            continue
        atoms.append(conv.Atom(name, xyz))


def load_pdb():
    banner("Cartesian → Internal: load a PDB file")
    print("  Type the path to a PDB file, or drag the file into this window.\n")
    raw = ask("PDB file (Enter to go back):")
    if not raw:
        return None
    path = resolve_input_path(raw)
    print()
    options = [("1", "Backbone atoms only (N, CA, C, O)"), ("2", "All atoms")]
    menu(options)
    backbone_only = choose(options, "Which atoms?") == "1"
    return io.read_pdb(path, backbone_only=backbone_only)


def cartesian_flow():
    while True:
        banner("Cartesian → Internal coordinates")
        print("  How would you like to enter the Cartesian coordinates?\n")
        options = [
            ("1", "Type them in", "x y z for the N, CA, C and O of each residue"),
            ("2", "Load a PDB file", "any .pdb file with atomic coordinates"),
            ("B", "Back to main menu"),
        ]
        menu(options)
        choice = choose(options)
        if choice == "b":
            return
        try:
            if choice == "1":
                atoms = type_cartesian()
            else:
                atoms = load_pdb()
        except (OSError, ValueError) as exc:
            print(bad("\n    Could not load the coordinates: {}".format(exc)))
            pause()
            continue
        if not atoms:
            continue

        banner("Cartesian → Internal: review")
        print("  {} atoms ready to convert:\n".format(len(atoms)))
        print_cartesian_table(atoms, limit=PREVIEW_ROWS)
        actions = [("C", "Convert to internal coordinates"), ("R", "Re-enter coordinates"), ("B", "Back to main menu")]
        menu(actions)
        action = choose(actions)
        if action == "b":
            return
        if action == "r":
            continue
        if show_cartesian_results(atoms) == "menu":
            return


def show_cartesian_results(atoms):
    try:
        entries = conv.cartesian_to_zmatrix(atoms)
        rebuilt = conv.zmatrix_to_cartesian(entries)
        torsions = conv.backbone_torsions(atoms)
        # Distances are the same for a mirror image, so the signed angles are checked too.
        matched = conv.require_internal_match(entries, rebuilt)
        if len(atoms) <= DISTANCE_CHECK_LIMIT:
            check = ("distances", conv.require_distance_match([a.xyz for a in atoms], rebuilt))
        else:
            check = ("internal", matched)
    except ValueError as exc:
        print(bad("\n    Conversion failed: {}".format(exc)))
        pause()
        return "again"

    is_protein = has_torsions(torsions)
    title = "Internal coordinates generated by {} ({})".format(STUDENT_NAME, STUDENT_ID)

    def show_zmatrix():
        print("  " + bold("Full Z-matrix (every atom)") + "\n")
        print_zmatrix_table(entries, atoms)
        print(muted("  Each atom is placed by a bond length to its bonded partner, an angle with"))
        print(muted("  that partner's partner, and a dihedral with the next atom along the chain.\n"))

    if is_protein:
        banner("Result: internal coordinates per residue")
        print_torsion_table(torsions)
        print(muted("  phi φ = C(i-1)-N-CA-C    psi ψ = N-CA-C-N(i+1)    omega ω = CA-C-N(i+1)-CA(i+1)"))
        print(muted("  Oxygen: C=O bond length, CA-C-O bond angle and N-CA-C-O dihedral."))
        print(muted("  '-' means the angle needs a neighbouring residue that is not there.\n"))
    else:
        banner("Result: internal coordinates (Z-matrix)")
        print(muted("  No protein backbone (N, CA, C) found, so showing the general Z-matrix.\n"))
        show_zmatrix()

    if check[0] == "distances":
        print(good("  ✓ Check: rebuilding Cartesian coordinates from these internal coordinates"))
        print(good("    reproduces all {} interatomic distances (largest deviation {:.1e} Å).".format(
            len(atoms) * (len(atoms) - 1) // 2, check[1])))
    else:
        bond_err, angle_err, dihedral_err = check[1]
        print(good("  ✓ Check: rebuilt coordinates reproduce every internal coordinate"))
        print(good("    (largest error {:.1e} Å, {:.1e}°, {:.1e}°).".format(bond_err, angle_err, dihedral_err)))
    print()

    save_zmatrix = ("X", "Save the full Z-matrix (.zmat)", lambda: save_prompt(
        "zmatrix_{}.zmat".format(timestamp()),
        lambda path: io.write_zmatrix(entries, path, title),
    ))
    save_amber = ("I", "Save AMBER internal coordinates (.in)", lambda: save_prompt(
        "internal_{}.in".format(timestamp()),
        lambda path: io.write_amber_internal(entries, path),
    ))
    if not is_protein:
        return results_menu([save_zmatrix, save_amber])

    def write_residue_table(path):
        with open(path, "w") as fh:
            fh.write(io.residue_table_csv(torsions))

    return results_menu([
        ("S", "Save the residue table (.csv, opens in Excel)", lambda: save_prompt(
            "residues_{}.csv".format(timestamp()), write_residue_table)),
        ("Z", "Show the full Z-matrix (every atom)", show_zmatrix),
        save_zmatrix,
        save_amber,
    ])


def results_menu(actions):
    options = [(key, label) for key, label, _ in actions] + [
        ("N", "New conversion"), ("M", "Main menu")]
    handlers = {key.lower(): handler for key, _, handler in actions}
    menu(options)
    while True:
        choice = choose(options, "What next?")
        if choice in handlers:
            handlers[choice]()
            menu(options)
        elif choice == "n":
            return "again"
        else:
            return "menu"


# ---------------------------------------------------------------------------
# Internal -> Cartesian
# ---------------------------------------------------------------------------

def internal_flow():
    while True:
        banner("Internal → Cartesian coordinates")
        print("  Which internal coordinates would you like to enter?\n")
        options = [
            ("1", "Per residue: phi, psi, omega and oxygen (recommended)",
             "φ ψ ω plus the CA-C-O and N-CA-C-O angles of each residue's carbonyl oxygen"),
            ("2", "Z-matrix (advanced)", "bond length, bond angle and dihedral for every atom"),
            ("B", "Back to main menu"),
        ]
        menu(options)
        choice = choose(options)
        if choice == "b":
            return
        result = backbone_flow() if choice == "1" else zmatrix_flow()
        if result == "menu":
            return


def type_zmatrix():
    banner("Internal → Cartesian: enter a Z-matrix")
    print("  Enter one atom per line. Lengths in Å, angles in degrees.\n")
    print("    Atom 1     NAME")
    print("    Atom 2     NAME  bond")
    print("    Atom 3     NAME  bond  angle")
    print("    Atom 4+    NAME  bond  angle  dihedral\n")
    print(muted("  By default the bond is to the previous atom, the angle is with the one"))
    print(muted("  before that, and the dihedral with the one before that. To pick the"))
    print(muted("  reference atoms yourself, put the atom number before each value:"))
    print(muted("    CB  2 1.530  1 110.5  3 -122.6\n"))
    print(muted("  Example (start of a protein backbone):"))
    print(muted("    N  /  CA 1.458  /  C 1.525 111.2  /  N 1.329 116.2 -47\n"))
    print(muted("  Commands: press Enter (or type 'done') to finish, 'undo', 'cancel'.\n"))

    entries = []
    while True:
        i = len(entries)
        raw = ask("Atom {} [{}]:".format(i + 1, io.zmatrix_hint(i)))
        command = raw.lower()
        if command in ("", "done"):
            if len(entries) >= 2:
                return entries
            print(warn("    Enter at least 2 atoms."))
            continue
        if command == "undo":
            if entries:
                removed = entries.pop()
                print(muted("    Removed atom {} ({}).".format(len(entries) + 1, removed.name)))
            continue
        if command in ("cancel", "b", "back"):
            return None
        try:
            entry = io.parse_zmatrix_line(raw, i)
            conv.validate_zmatrix(entries + [entry])
        except ValueError as exc:
            print(bad("    {}".format(exc)))
            continue
        entries.append(entry)


def load_zmatrix():
    banner("Internal → Cartesian: load a Z-matrix file")
    print("  Type the path to a .zmat or AMBER .in file, or drag the file into this window.")
    print(muted("  Files saved from the Cartesian → Internal screen can be loaded here.\n"))
    raw = ask("Z-matrix file (Enter to go back):")
    if not raw:
        return None
    return io.read_zmatrix(resolve_input_path(raw))


def zmatrix_flow():
    while True:
        banner("Internal → Cartesian: Z-matrix")
        print("  How would you like to enter the internal coordinates?\n")
        options = [
            ("1", "Type them in", "one atom per line:  NAME  bond  angle  dihedral"),
            ("2", "Load a Z-matrix file (.zmat or .in)", "saved .zmat files, or an AMBER internal-coordinate file"),
            ("B", "Back"),
        ]
        menu(options)
        choice = choose(options)
        if choice == "b":
            return "back"
        try:
            if choice == "1":
                entries = type_zmatrix()
            else:
                entries = load_zmatrix()
            if entries:
                conv.validate_zmatrix(entries)
        except (OSError, ValueError) as exc:
            print(bad("\n    Could not load the Z-matrix: {}".format(exc)))
            pause()
            continue
        if not entries:
            continue

        banner("Internal → Cartesian: review")
        print("  {} atoms ready to convert:\n".format(len(entries)))
        print_zmatrix_table(entries, limit=PREVIEW_ROWS)
        actions = [("C", "Convert to Cartesian coordinates"), ("R", "Re-enter values"), ("B", "Back")]
        menu(actions)
        action = choose(actions)
        if action == "b":
            return "back"
        if action == "r":
            continue
        if show_zmatrix_results(entries) == "menu":
            return "menu"


def show_zmatrix_results(entries):
    try:
        coords = conv.zmatrix_to_cartesian(entries)
        bond_err, angle_err, dihedral_err = conv.require_internal_match(entries, coords)
    except ValueError as exc:
        print(bad("\n    Conversion failed: {}".format(exc)))
        pause()
        return "again"
    atoms = io.assign_residues([conv.Atom(e.name, xyz) for e, xyz in zip(entries, coords)])

    banner("Result: Cartesian coordinates")
    print_cartesian_table(atoms)
    print(muted("  Atom 1 is placed at the origin, atom 2 on the +x axis and atom 3 in the"))
    print(muted("  xy-plane; internal coordinates do not fix the overall position/orientation.\n"))

    torsions = conv.backbone_torsions(atoms)
    if has_torsions(torsions):
        print("  " + bold("Per-residue internal coordinates (φ, ψ, ω and oxygen)") + "\n")
        print_torsion_table(torsions)

    print(good("  ✓ Check: measured back from these coordinates, every bond, angle and"))
    print(good("    dihedral matches your input (largest error {:.1e} Å, {:.1e}°, {:.1e}°).\n".format(
        bond_err, angle_err, dihedral_err)))

    return results_menu([
        ("S", "Save as a PDB file", lambda: save_prompt(
            "structure_{}.pdb".format(timestamp()),
            lambda path: io.write_pdb(atoms, path, "Built from a Z-matrix by {} ({})".format(STUDENT_NAME, STUDENT_ID)),
        )),
    ])


PRESETS = [
    ("1", "alpha-helix", "α-helix", "phi -57, psi -47, omega 180"),
    ("2", "beta-strand", "β-strand", "phi -139, psi 135, omega 180"),
    ("3", "polyproline-II", "Polyproline II helix", "phi -75, psi 145, omega 180"),
    ("4", "extended", "Fully extended chain", "phi 180, psi 180, omega 180"),
]


def ask_numbers(prompt, names, defaults):
    """Ask for len(names) numbers on one line; Enter accepts the defaults. None = go back."""
    default_text = " ".join("{:g}".format(d) for d in defaults)
    while True:
        raw = ask("{} [{}] (Enter = {}):".format(prompt, "  ".join(names), default_text))
        if raw.lower() in ("cancel", "b", "back"):
            return None
        tokens = raw.replace(",", " ").split()
        if not tokens:
            return list(defaults)
        if len(tokens) != len(names):
            print(bad("    Please enter {} value(s): {}".format(len(names), "  ".join(names))))
            continue
        try:
            return [float(t) for t in tokens]
        except ValueError:
            print(bad("    Please enter numbers only."))


def ask_residue_names():
    count = ask_int("Number of residues (1-500, 'b' to go back):", 1, 500)
    if count is None:
        return None
    while True:
        seq = ask("Sequence in one-letter code (optional, Enter = all ALA):")
        try:
            return io.sequence_to_residues(seq, count)
        except ValueError as exc:
            print(bad("    {}".format(exc)))


def type_residues():
    banner("Internal → Cartesian: type φ, ψ, ω and oxygen per residue")
    res_names = ask_residue_names()
    if res_names is None:
        return None
    count = len(res_names)
    print()
    print("  Each residue takes two lines: its torsions (φ ψ ω), then its oxygen")
    print("  (the CA-C-O angle and the N-CA-C-O dihedral). Press Enter to accept the")
    print("  value in brackets. Residue 1 has no φ and the last residue has no ω,")
    print("  because those angles need a neighbouring residue.")
    print(muted("  The default N-CA-C-O is ψ + 180°, which keeps O in the peptide plane."))
    print(muted("  Type 'b' to go back.\n"))

    residues = []
    for i, name in enumerate(res_names):
        label = "Residue {} {}".format(i + 1, name)
        names = (["phi"] if i > 0 else []) + ["psi"] + (["omega"] if i < count - 1 else [])
        defaults = {"phi": -57.0, "psi": -47.0, "omega": 180.0}
        values = ask_numbers(label + " torsions", names, [defaults[n] for n in names])
        if values is None:
            return None
        chosen = dict(zip(names, values))
        psi = chosen["psi"]
        oxygen = ask_numbers(" " * len(label) + " oxygen  ", ["CA-C-O", "N-CA-C-O"],
                             [conv.ANGLE_CA_C_O, round(conv.oxygen_dihedral_from_psi(psi), 2)])
        if oxygen is None:
            return None
        residues.append(conv.ResidueTorsions(
            name, i + 1, "A", chosen.get("phi"), psi, chosen.get("omega"),
            conv.BOND_C_O, oxygen[0], oxygen[1]))
        print()
    return residues


def preset_residues():
    banner("Internal → Cartesian: secondary-structure preset")
    res_names = ask_residue_names()
    if res_names is None:
        return None
    print()
    options = [(key, label, desc) for key, _, label, desc in PRESETS]
    menu(options)
    choice = choose(options, "Backbone conformation:")
    phi, psi, omega = conv.SECONDARY_STRUCTURE_PRESETS[PRESETS[int(choice) - 1][1]]
    return [conv.ResidueTorsions(name, i + 1, "A", phi, psi, omega) for i, name in enumerate(res_names)]


def load_residue_table():
    banner("Internal → Cartesian: load a residue table")
    print("  Type the path to a .csv file, or drag the file into this window.")
    print(muted("  Columns: " + ", ".join(io.RESIDUE_COLUMNS)))
    print(muted("  Residue tables saved from the Cartesian → Internal screen can be loaded here.\n"))
    raw = ask("Residue table (Enter to go back):")
    if not raw:
        return None
    return io.read_residue_csv(resolve_input_path(raw))


def backbone_flow():
    while True:
        banner("Internal → Cartesian: φ, ψ, ω and oxygen per residue")
        print("  Builds the backbone atoms N, CA, C and O of every residue. Bond lengths")
        print("  and the N-CA-C, CA-C-N and C-N-CA angles use ideal values (Engh & Huber):")
        print(muted("    N-CA {} Å, CA-C {} Å, C-N {} Å, C=O {} Å".format(
            conv.BOND_N_CA, conv.BOND_CA_C, conv.BOND_C_N, conv.BOND_C_O)))
        print(muted("    N-CA-C {}°, CA-C-N {}°, C-N-CA {}°\n".format(
            conv.ANGLE_N_CA_C, conv.ANGLE_CA_C_N, conv.ANGLE_C_N_CA)))
        options = [
            ("1", "Type them in", "φ ψ ω, then the oxygen angles, for each residue"),
            ("2", "Use a secondary-structure preset", "α-helix, β-strand, polyproline II or extended"),
            ("3", "Load a residue table (.csv)", "files saved from Cartesian → Internal"),
            ("B", "Back"),
        ]
        menu(options)
        choice = choose(options)
        if choice == "b":
            return "back"
        try:
            if choice == "1":
                residues = type_residues()
            elif choice == "2":
                residues = preset_residues()
            else:
                residues = load_residue_table()
            if residues:
                residues = conv.complete_residues(residues)
        except (OSError, ValueError) as exc:
            print(bad("\n    Could not use these values: {}".format(exc)))
            pause()
            continue
        if not residues:
            continue

        banner("Internal → Cartesian: review")
        print("  {} residues ready to convert:\n".format(len(residues)))
        print_torsion_table(residues, limit=PREVIEW_ROWS)
        print(muted("  The first residue has no φ and the last has no ω: those angles need"))
        print(muted("  atoms from a neighbouring residue that does not exist.\n"))
        actions = [("C", "Convert to Cartesian coordinates"), ("R", "Re-enter values"), ("B", "Back")]
        menu(actions)
        action = choose(actions)
        if action == "b":
            return "back"
        if action == "r":
            continue
        if show_backbone_results(residues) == "menu":
            return "menu"


def show_backbone_results(residues):
    try:
        atoms = conv.build_backbone(residues)
        worst = conv.require_torsion_match(atoms, residues)
    except ValueError as exc:
        print(bad("\n    Conversion failed: {}".format(exc)))
        pause()
        return "again"

    banner("Result: Cartesian coordinates of the backbone")
    print_cartesian_table(atoms)
    print(good("  ✓ Check: φ, ψ, ω and the oxygen angles measured back from these"))
    print(good("    coordinates match your input (largest error {:.1e}°).\n".format(worst)))
    print(muted("  Tip: save as PDB and open it in PyMOL, ChimeraX or VMD to view it.\n"))

    return results_menu([
        ("S", "Save as a PDB file", lambda: save_prompt(
            "backbone_{}.pdb".format(timestamp()),
            lambda path: io.write_pdb(atoms, path, "Backbone built by {} ({})".format(STUDENT_NAME, STUDENT_ID)),
        )),
    ])


# ---------------------------------------------------------------------------
# Help and main menu
# ---------------------------------------------------------------------------

HELP_TEXT = """\
  CARTESIAN COORDINATES
    Every atom has an (x, y, z) position in Ångström, as stored in a PDB
    file. They describe both the shape of the molecule and where it sits
    and how it is oriented in space.

  INTERNAL COORDINATES OF A PROTEIN (PER RESIDUE)
    Each residue is described by its backbone torsion angles:
      phi   (φ) = C(i-1) - N(i)  - CA(i)  - C(i)
      psi   (ψ) = N(i)   - CA(i) - C(i)   - N(i+1)
      omega (ω) = CA(i)  - C(i)  - N(i+1) - CA(i+1)   (~180° for trans peptides)
    and by the position of its carbonyl oxygen:
      C=O bond length (~1.23 Å), CA-C-O angle (~120.5°), N-CA-C-O dihedral
      (about ψ + 180°, because O sits in the peptide plane opposite N(i+1)).
    Typical values: α-helix φ -57 ψ -47, β-strand φ -139 ψ 135.
    Other bond lengths and angles are almost constant, so ideal values
    (Engh & Huber) are used when building Cartesian coordinates.

  Z-MATRIX (ADVANCED)
    The general form: each atom is described relative to atoms placed
    before it by a bond length (Å), a bond angle (°) and a dihedral (°).
    Internal coordinates describe only the shape, so converting back to
    Cartesian puts atom 1 at the origin, atom 2 on the x-axis and atom 3
    in the xy-plane.

  SIGN CONVENTION
    Dihedrals follow IUPAC: looking along the middle bond, positive means
    the front atom turns clockwise to eclipse the back atom. Range -180..180.

  TIPS
    Type 'q' at any prompt to quit. Saved residue tables (.csv) and
    Z-matrix files (.zmat) and AMBER internal-coordinate files (.in) can be
    loaded back on the Internal → Cartesian screen.
"""


def show_help():
    banner("Help: coordinate systems")
    print(HELP_TEXT)
    pause()


def main_menu():
    while True:
        banner()
        print("  Which coordinates would you like to enter?\n")
        options = [
            ("1", "Cartesian coordinates", "x, y, z of each atom  →  φ, ψ, ω and oxygen angles per residue"),
            ("2", "Internal coordinates", "φ, ψ, ω and oxygen angles per residue  →  x, y, z of each atom"),
            ("H", "Help: what are these coordinates?"),
            ("Q", "Quit"),
        ]
        menu(options)
        choice = choose(options)
        if choice == "1":
            cartesian_flow()
        elif choice == "2":
            internal_flow()
        elif choice == "h":
            show_help()


def main():
    try:
        main_menu()
    except (KeyboardInterrupt, QuitProgram):
        pass
    print("\n  Goodbye!\n")


if __name__ == "__main__":
    main()
