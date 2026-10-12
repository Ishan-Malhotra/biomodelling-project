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
EXAMPLES_DIR = os.path.join(HERE, "examples")
OUTPUT_DIR = os.path.join(HERE, "output")
EXAMPLE_PDB = os.path.join(EXAMPLES_DIR, "crambin_fragment.pdb")
EXAMPLE_ZMAT = os.path.join(EXAMPLES_DIR, "crambin_fragment.zmat")

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
    row("Cartesian (x, y, z)  <->  Internal (bond, angle, dihedral)", muted, center=True)
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


def print_table(headers, rows, aligns, limit=None):
    shown = rows if limit is None or len(rows) <= limit else rows[:limit]
    widths = [max([len(h)] + [len(r[i]) for r in shown]) for i, h in enumerate(headers)]

    def fmt(cells):
        return "  " + "  ".join(
            c.rjust(w) if a == "r" else c.ljust(w) for c, w, a in zip(cells, widths, aligns)
        )

    print(bold(fmt(headers)))
    print(muted("  " + "  ".join("─" * w for w in widths)))
    for r in shown:
        print(fmt(r))
    if len(shown) < len(rows):
        print(muted("  ... and {} more rows".format(len(rows) - len(shown))))
    print()


def print_cartesian_table(atoms, limit=None):
    rows = [
        [str(i), a.name, "{}{}".format(a.res_name, a.res_seq), fnum(a.xyz[0], 3), fnum(a.xyz[1], 3), fnum(a.xyz[2], 3)]
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
    return any(t.phi is not None or t.psi is not None for t in torsions)


def print_torsion_table(torsions, limit=None):
    rows = [
        ["{}{}".format(t.res_name, t.res_seq), t.chain, fnum(t.phi, 1), fnum(t.psi, 1),
         fnum(t.omega, 1), conv.ramachandran_region(t.phi, t.psi)]
        for t in torsions
    ]
    print_table(["Residue", "Chain", "phi φ (°)", "psi ψ (°)", "omega ω (°)", "Region"], rows, "llrrrl", limit)


# ---------------------------------------------------------------------------
# Cartesian -> internal
# ---------------------------------------------------------------------------

def type_cartesian():
    banner("Cartesian → Internal: enter coordinates")
    print("  Enter one atom per line as   NAME  X  Y  Z   (in Ångström).")
    print("  Use PDB atom names (N, CA, C, O, CB, ...) so residues are recognised.\n")
    print(muted("  Example:   N   17.047  14.099  3.625"))
    print(muted("             CA  16.967  12.784  4.338"))
    print(muted("             C   15.685  12.755  5.133\n"))
    print(muted("  Commands: press Enter (or type 'done') to finish, 'undo' to remove"))
    print(muted("            the last atom, 'cancel' to go back.\n"))

    atoms = []
    while True:
        raw = ask("Atom {}:".format(len(atoms) + 1))
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
            name, xyz = io.parse_cartesian_line(raw, "A{}".format(len(atoms) + 1))
        except ValueError as exc:
            print(bad("    {}".format(exc)))
            continue
        atoms.append(conv.Atom(name, xyz))


def load_pdb():
    banner("Cartesian → Internal: load a PDB file")
    print("  Type the path to a PDB file, or drag the file into this window.")
    print(muted("  Example: examples/1crn.pdb  (crambin, 46 residues)\n"))
    raw = ask("PDB file (Enter to go back):")
    if not raw:
        return None
    path = resolve_input_path(raw)
    print()
    options = [("1", "Backbone atoms only (N, CA, C)"), ("2", "All atoms")]
    menu(options)
    backbone_only = choose(options, "Which atoms?") == "1"
    return io.read_pdb(path, backbone_only=backbone_only)


def cartesian_flow():
    while True:
        banner("Cartesian → Internal coordinates")
        print("  How would you like to enter the Cartesian coordinates?\n")
        options = [
            ("1", "Type them in", "one atom per line:  NAME  X  Y  Z"),
            ("2", "Load a PDB file", "e.g. examples/1crn.pdb"),
            ("3", "Use the example", "backbone of crambin residues 1-4 (12 atoms)"),
            ("B", "Back to main menu"),
        ]
        menu(options)
        choice = choose(options)
        if choice == "b":
            return
        try:
            if choice == "1":
                atoms = type_cartesian()
            elif choice == "2":
                atoms = load_pdb()
            else:
                atoms = io.read_pdb(EXAMPLE_PDB)
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
    except ValueError as exc:
        print(bad("\n    Conversion failed at {}".format(exc)))
        pause()
        return "again"
    torsions = conv.backbone_torsions(atoms)

    banner("Result: internal coordinates (Z-matrix)")
    print_zmatrix_table(entries, atoms)
    print(muted("  Each atom is placed by a bond length to its bonded partner, an angle with"))
    print(muted("  that partner's partner, and a dihedral with the next atom along the chain.\n"))

    if has_torsions(torsions):
        print("  " + bold("Protein backbone torsion angles") + "\n")
        print_torsion_table(torsions)

    if len(atoms) <= DISTANCE_CHECK_LIMIT:
        deviation = conv.max_distance_deviation([a.xyz for a in atoms], rebuilt)
        print(good("  ✓ Check: rebuilding Cartesian coordinates from this Z-matrix reproduces all"))
        print(good("    {} interatomic distances (largest deviation {:.1e} Å).".format(
            len(atoms) * (len(atoms) - 1) // 2, deviation)))
    else:
        bond_err, angle_err, dihedral_err = conv.zmatrix_errors(entries, rebuilt)
        print(good("  ✓ Check: rebuilt coordinates reproduce every internal coordinate"))
        print(good("    (largest error {:.1e} Å, {:.1e}°, {:.1e}°).".format(bond_err, angle_err, dihedral_err)))
    print()

    return results_menu([
        ("S", "Save Z-matrix to a file", lambda: save_prompt(
            "zmatrix_{}.zmat".format(timestamp()),
            lambda path: io.write_zmatrix(entries, path, "Z-matrix generated by {} ({})".format(STUDENT_NAME, STUDENT_ID)),
        )),
    ])


def results_menu(save_actions):
    options = [(key, label) for key, label, _ in save_actions] + [
        ("N", "New conversion"), ("M", "Main menu")]
    handlers = {key.lower(): handler for key, _, handler in save_actions}
    menu(options)
    while True:
        choice = choose(options, "What next?")
        if choice in handlers:
            handlers[choice]()
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
            ("1", "Z-matrix", "bond lengths, bond angles and dihedral angles, atom by atom"),
            ("2", "Protein backbone torsions", "phi / psi / omega per residue, with ideal bond geometry"),
            ("B", "Back to main menu"),
        ]
        menu(options)
        choice = choose(options)
        if choice == "b":
            return
        result = zmatrix_flow() if choice == "1" else backbone_flow()
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
    print("  Type the path to a .zmat file, or drag the file into this window.")
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
            ("2", "Load a Z-matrix file (.zmat)", "e.g. examples/crambin_fragment.zmat"),
            ("3", "Use the example", "backbone of crambin residues 1-4 (12 atoms)"),
            ("B", "Back"),
        ]
        menu(options)
        choice = choose(options)
        if choice == "b":
            return "back"
        try:
            if choice == "1":
                entries = type_zmatrix()
            elif choice == "2":
                entries = load_zmatrix()
            else:
                entries = io.read_zmatrix(EXAMPLE_ZMAT)
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
    except ValueError as exc:
        print(bad("\n    Conversion failed at {}".format(exc)))
        pause()
        return "again"
    atoms = io.assign_residues([conv.Atom(e.name, xyz) for e, xyz in zip(entries, coords)])
    bond_err, angle_err, dihedral_err = conv.zmatrix_errors(entries, coords)

    banner("Result: Cartesian coordinates")
    print_cartesian_table(atoms)
    print(muted("  Atom 1 is placed at the origin, atom 2 on the +x axis and atom 3 in the"))
    print(muted("  xy-plane; internal coordinates do not fix the overall position/orientation.\n"))

    torsions = conv.backbone_torsions(atoms)
    if has_torsions(torsions):
        print("  " + bold("Protein backbone torsion angles") + "\n")
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


def ask_torsions(index, count, res_name):
    """Ask for the torsions residue `index` actually needs; returns (phi, psi, omega)."""
    needs = []
    if index > 0:
        needs.append("phi")
    if index < count - 1:
        needs += ["psi", "omega"]
    defaults = {"phi": -57.0, "psi": -47.0, "omega": 180.0}
    default_text = " ".join("{:g}".format(defaults[n]) for n in needs)
    while True:
        raw = ask("Residue {} {} [{}] (Enter = {}):".format(index + 1, res_name, " ".join(needs), default_text))
        if raw.lower() in ("cancel", "b", "back"):
            return None
        tokens = raw.replace(",", " ").split()
        if not tokens:
            values = [defaults[n] for n in needs]
        elif len(tokens) == len(needs):
            try:
                values = [float(t) for t in tokens]
            except ValueError:
                print(bad("    Please enter numbers only."))
                continue
        else:
            print(bad("    Please enter {} value(s): {}".format(len(needs), " ".join(needs))))
            continue
        chosen = dict(zip(needs, values))
        return chosen.get("phi"), chosen.get("psi"), chosen.get("omega")


def backbone_flow():
    while True:
        banner("Internal → Cartesian: protein backbone")
        print("  Build a backbone (N, CA, C atoms) from its torsion angles. Bond lengths")
        print("  and angles use ideal values (Engh & Huber):")
        print(muted("    N-CA {} Å, CA-C {} Å, C-N {} Å".format(conv.BOND_N_CA, conv.BOND_CA_C, conv.BOND_C_N)))
        print(muted("    N-CA-C {}°, CA-C-N {}°, C-N-CA {}°\n".format(
            conv.ANGLE_N_CA_C, conv.ANGLE_CA_C_N, conv.ANGLE_C_N_CA)))

        count = ask_int("Number of residues (1-500, 'b' to go back):", 1, 500)
        if count is None:
            return "back"
        while True:
            seq = ask("Sequence in one-letter code (optional, Enter = all ALA):")
            try:
                res_names = io.sequence_to_residues(seq, count)
                break
            except ValueError as exc:
                print(bad("    {}".format(exc)))

        print()
        options = [(key, label, desc) for key, _, label, desc in PRESETS]
        options.append(("5", "Enter phi / psi / omega for each residue"))
        menu(options)
        choice = choose(options, "Backbone conformation:")
        if choice == "5":
            print()
            per_residue = []
            for i, name in enumerate(res_names):
                values = ask_torsions(i, count, name)
                if values is None:
                    break
                per_residue.append(values)
            if len(per_residue) < count:
                continue
            phis, psis, omegas = (list(column) for column in zip(*per_residue))
        else:
            phi, psi, omega = conv.SECONDARY_STRUCTURE_PRESETS[PRESETS[int(choice) - 1][1]]
            phis, psis, omegas = [phi] * count, [psi] * count, [omega] * count

        phis[0] = None
        psis[-1] = None
        omegas[-1] = None
        torsions = [conv.ResidueTorsions(name, i + 1, "A", phis[i], psis[i], omegas[i])
                    for i, name in enumerate(res_names)]

        banner("Internal → Cartesian: review")
        print("  {} residues ready to convert:\n".format(count))
        print_torsion_table(torsions, limit=PREVIEW_ROWS)
        print(muted("  The first residue has no phi and the last has no psi/omega: those"))
        print(muted("  angles need atoms from a neighbouring residue that does not exist.\n"))
        actions = [("C", "Convert to Cartesian coordinates"), ("R", "Re-enter values"), ("B", "Back")]
        menu(actions)
        action = choose(actions)
        if action == "b":
            return "back"
        if action == "r":
            continue
        if show_backbone_results(res_names, phis, psis, omegas) == "menu":
            return "menu"


def show_backbone_results(res_names, phis, psis, omegas):
    try:
        atoms = conv.build_backbone(res_names, phis, psis, omegas)
    except ValueError as exc:
        print(bad("\n    Conversion failed at {}".format(exc)))
        pause()
        return "again"

    worst = conv.backbone_torsion_error(atoms, phis, psis, omegas)

    banner("Result: Cartesian coordinates of the backbone")
    print_cartesian_table(atoms)
    print(good("  ✓ Check: phi/psi/omega measured back from these coordinates match your"))
    print(good("    input (largest error {:.1e}°).\n".format(worst)))
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

  INTERNAL COORDINATES (Z-MATRIX)
    Each atom is described relative to atoms placed before it:
      bond length    distance to one earlier atom              (Å)
      bond angle     angle made with a second earlier atom     (°)
      dihedral       twist about the bond, using a third atom  (°)
    They describe only the shape, so converting back to Cartesian puts
    atom 1 at the origin, atom 2 on the x-axis and atom 3 in the xy-plane.

  PROTEIN BACKBONE TORSIONS
    phi   (φ) = C(i-1) - N(i)  - CA(i)  - C(i)
    psi   (ψ) = N(i)   - CA(i) - C(i)   - N(i+1)
    omega (ω) = CA(i)  - C(i)  - N(i+1) - CA(i+1)   (~180° for trans peptides)
    Typical values: α-helix φ -57 ψ -47, β-strand φ -139 ψ 135.

  SIGN CONVENTION
    Dihedrals follow IUPAC: looking along the middle bond, positive means
    the front atom turns clockwise to eclipse the back atom. Range -180..180.

  TIPS
    Type 'q' at any prompt to quit. Saved Z-matrix files (.zmat) can be
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
            ("1", "Cartesian coordinates", "x, y, z of each atom  →  converted to internal coordinates"),
            ("2", "Internal coordinates", "bond lengths, angles, dihedrals  →  converted to Cartesian"),
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
