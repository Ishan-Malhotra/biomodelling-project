import math
import os
import random
import tempfile
import unittest

from coordconv import conversions as conv
from coordconv import geometry as geo
from coordconv import io_utils as io


def helix_residues(n, sequence=""):
    phi, psi, omega = conv.SECONDARY_STRUCTURE_PRESETS["alpha-helix"]
    names = io.sequence_to_residues(sequence, n)
    return [conv.ResidueTorsions(name, i + 1, "A", phi, psi, omega) for i, name in enumerate(names)]


def helix_atoms(n, sequence=""):
    return conv.build_backbone(helix_residues(n, sequence))


class GeometryTests(unittest.TestCase):
    def test_dihedral_sign_convention(self):
        a, b, c = (1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)
        self.assertAlmostEqual(geo.dihedral(a, b, c, (1.0, 1.0, 0.0)), 0.0)
        self.assertAlmostEqual(abs(geo.dihedral(a, b, c, (-1.0, 1.0, 0.0))), 180.0)
        self.assertAlmostEqual(geo.dihedral(a, b, c, (0.0, 1.0, -1.0)), 90.0)
        self.assertAlmostEqual(geo.dihedral(a, b, c, (0.0, 1.0, 1.0)), -90.0)

    def test_place_atom_reproduces_internal_coordinates(self):
        rng = random.Random(1)
        for _ in range(200):
            a, b, c = [tuple(rng.uniform(-5, 5) for _ in range(3)) for _ in range(3)]
            bond, angle, torsion = rng.uniform(0.9, 2.0), rng.uniform(20, 170), rng.uniform(-180, 180)
            d = geo.place_atom(a, b, c, bond, angle, torsion)
            self.assertAlmostEqual(geo.distance(c, d), bond, places=9)
            self.assertAlmostEqual(geo.bond_angle(b, c, d), angle, places=7)
            self.assertAlmostEqual(geo.angle_difference(geo.dihedral(a, b, c, d), torsion), 0.0, places=7)


class RoundTripTests(unittest.TestCase):
    def test_built_backbone_round_trip(self):
        atoms = helix_atoms(12, "AGKV")
        entries = conv.cartesian_to_zmatrix(atoms)
        rebuilt = conv.zmatrix_to_cartesian(entries)
        self.assertLess(conv.max_distance_deviation([a.xyz for a in atoms], rebuilt), 1e-9)

    def test_zmatrix_round_trip_through_file(self):
        atoms = helix_atoms(8)
        entries = conv.cartesian_to_zmatrix(atoms)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.zmat")
            io.write_zmatrix(entries, path)
            loaded = io.read_zmatrix(path)
        rebuilt = conv.zmatrix_to_cartesian(loaded)
        # The file stores 4 decimals for bonds and 3 for angles.
        self.assertLess(conv.max_distance_deviation([a.xyz for a in atoms], rebuilt), 0.05)

    def test_references_follow_bonds(self):
        atoms = helix_atoms(10)
        entries = conv.cartesian_to_zmatrix(atoms)
        for i, e in enumerate(entries[1:], start=1):
            self.assertLess(e.bond, 2.1, "atom {} {} bonded to {}".format(i + 1, e.name, entries[e.bond_ref].name))


class ProteinTests(unittest.TestCase):
    def test_helix_ramachandran_regions(self):
        torsions = conv.backbone_torsions(helix_atoms(20))
        self.assertEqual(len(torsions), 20)
        self.assertIsNone(torsions[0].phi)
        self.assertIsNone(torsions[-1].psi)
        for t in torsions[1:-1]:
            self.assertEqual(conv.ramachandran_region(t.phi, t.psi), "alpha-R")
            self.assertGreater(abs(t.omega), 160)

    def test_backbone_zmatrix_dihedrals_are_phi_psi_omega(self):
        atoms = helix_atoms(6)
        entries = conv.cartesian_to_zmatrix(atoms)
        torsions = conv.backbone_torsions(atoms)
        self.assertEqual([e.name for e in entries[:4]], ["N", "CA", "C", "O"])
        for r in range(1, len(torsions)):
            n, ca, c, o = entries[4 * r: 4 * r + 4]
            self.assertAlmostEqual(n.dihedral, torsions[r - 1].psi, places=6)
            self.assertAlmostEqual(ca.dihedral, torsions[r - 1].omega, places=6)
            self.assertAlmostEqual(c.dihedral, torsions[r].phi, places=6)
            self.assertAlmostEqual(o.dihedral, torsions[r].o_dihedral, places=6)

    def test_built_oxygen_lies_in_peptide_plane(self):
        for t in conv.backbone_torsions(helix_atoms(10)):
            self.assertAlmostEqual(t.o_bond, conv.BOND_C_O, places=9)
            self.assertAlmostEqual(t.o_angle, conv.ANGLE_CA_C_O, places=6)
            if t.psi is not None:
                self.assertLess(abs(geo.angle_difference(t.o_dihedral, t.psi + 180.0)), 1e-6)

    def test_build_backbone_reproduces_torsions_and_oxygen(self):
        rng = random.Random(7)
        n = 12
        residues = [
            conv.ResidueTorsions("ALA", i + 1, "A", rng.uniform(-180, 180), rng.uniform(-180, 180),
                                 rng.choice([180.0, -175.0, 5.0]), rng.uniform(1.2, 1.26),
                                 rng.uniform(115, 125), rng.uniform(-180, 180))
            for i in range(n)
        ]
        atoms = conv.build_backbone(residues)
        self.assertEqual([a.name for a in atoms[:8]], ["N", "CA", "C", "O"] * 2)
        self.assertEqual(len(atoms), 4 * n)
        for i, (t, want) in enumerate(zip(conv.backbone_torsions(atoms), residues)):
            if i > 0:
                self.assertAlmostEqual(geo.angle_difference(t.phi, want.phi), 0.0, places=6)
            if i < n - 1:
                self.assertAlmostEqual(geo.angle_difference(t.psi, want.psi), 0.0, places=6)
                self.assertAlmostEqual(geo.angle_difference(t.omega, want.omega), 0.0, places=6)
            self.assertAlmostEqual(t.o_bond, want.o_bond, places=9)
            self.assertAlmostEqual(t.o_angle, want.o_angle, places=6)
            self.assertAlmostEqual(geo.angle_difference(t.o_dihedral, want.o_dihedral), 0.0, places=6)
        self.assertLess(conv.backbone_torsion_error(atoms, residues), 1e-6)

    def test_measured_residues_rebuild_the_same_backbone(self):
        atoms = helix_atoms(10)
        rebuilt = conv.build_backbone(conv.backbone_torsions(atoms))
        self.assertEqual([a.name for a in rebuilt], [a.name for a in atoms])
        self.assertLess(conv.max_distance_deviation([a.xyz for a in atoms], [a.xyz for a in rebuilt]), 1e-9)

    def test_oxygen_defaults_and_missing_values(self):
        residues = conv.complete_residues([
            conv.ResidueTorsions("ALA", 1, "A", None, 120.0, 180.0),
            conv.ResidueTorsions("GLY", 2, "A", -60.0, -40.0, None),
        ])
        self.assertEqual((residues[0].o_bond, residues[0].o_angle), (conv.BOND_C_O, conv.ANGLE_CA_C_O))
        self.assertAlmostEqual(residues[0].o_dihedral, -60.0)
        self.assertAlmostEqual(residues[1].o_dihedral, 140.0)
        with self.assertRaises(ValueError):
            conv.complete_residues([conv.ResidueTorsions("ALA", 1, "A", None, None, None)])
        with self.assertRaises(ValueError):
            conv.complete_residues([conv.ResidueTorsions("ALA", 1, "A", None, 120.0, None),
                                    conv.ResidueTorsions("ALA", 2, "A", -60.0, 120.0, None)])

    def test_ideal_helix_rise_per_residue(self):
        atoms = helix_atoms(20)
        ca = [a.xyz for a in atoms if a.name == "CA"]
        # i -> i+3 CA distance in an alpha-helix is about 5.0-5.4 Angstrom.
        self.assertTrue(4.8 < geo.distance(ca[5], ca[8]) < 5.6)
        # In an alpha-helix C=O(i) hydrogen-bonds to N-H(i+4): O(i)...N(i+4) is about 2.9-3.1 Angstrom.
        o = [a.xyz for a in atoms if a.name == "O"]
        n = [a.xyz for a in atoms if a.name == "N"]
        self.assertTrue(2.6 < geo.distance(o[5], n[9]) < 3.4)


class ParsingTests(unittest.TestCase):
    def test_short_and_full_zmatrix_lines(self):
        short = io.parse_zmatrix_line("N 1.329 116.2 -47", 3)
        full = io.parse_zmatrix_line("N 3 1.329 2 116.2 1 -47", 3)
        self.assertEqual(short, full)
        self.assertEqual((short.bond_ref, short.angle_ref, short.dihedral_ref), (2, 1, 0))

    def test_bad_lines_are_rejected(self):
        with self.assertRaises(ValueError):
            io.parse_zmatrix_line("CA 1.5 110", 3)
        with self.assertRaises(ValueError):
            io.parse_cartesian_line("CA 1.0 abc 3.0", "A1")
        with self.assertRaises(ValueError):
            conv.validate_zmatrix([conv.ZEntry("N"), conv.ZEntry("CA", 0, -1.0)])

    def test_residue_table_round_trip(self):
        atoms = helix_atoms(8, "AGKV")
        self.assertEqual({a.name for a in atoms}, {"N", "CA", "C", "O"})
        torsions = conv.backbone_torsions(atoms)
        loaded = io.parse_residue_csv(io.residue_table_csv(torsions))
        self.assertEqual(len(loaded), 8)
        self.assertEqual((loaded[0].res_name, loaded[0].res_seq), ("ALA", 1))
        self.assertEqual(loaded[1].res_name, "GLY")
        self.assertIsNone(loaded[0].phi)
        self.assertIsNone(loaded[-1].omega)
        for a, b in zip(torsions, loaded):
            for field in ("phi", "psi", "omega", "o_angle", "o_dihedral"):
                x, y = getattr(a, field), getattr(b, field)
                self.assertTrue(x is None and y is None or abs(x - y) < 0.01)

    def test_residue_table_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            io.parse_residue_csv("residue,phi\nALA1,-60\n")
        with self.assertRaises(ValueError):
            io.parse_residue_csv("residue,phi,psi,omega\nALA1,-60,abc,180\n")

    def test_pdb_write_read(self):
        atoms = helix_atoms(6, "AGK")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pdb")
            io.write_pdb(atoms, path)
            again = io.read_pdb(path)
        self.assertEqual([a.name for a in atoms], [a.name for a in again])
        for a, b in zip(atoms, again):
            self.assertTrue(all(math.isclose(p, q, abs_tol=1e-3) for p, q in zip(a.xyz, b.xyz)))


if __name__ == "__main__":
    unittest.main()
