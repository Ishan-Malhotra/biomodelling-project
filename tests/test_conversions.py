import math
import os
import random
import tempfile
import unittest

from coordconv import conversions as conv
from coordconv import geometry as geo
from coordconv import io_utils as io

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
CRAMBIN = os.path.join(EXAMPLES, "1crn.pdb")


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
    def test_crambin_all_atoms_round_trip(self):
        atoms = io.read_pdb(CRAMBIN)
        entries = conv.cartesian_to_zmatrix(atoms)
        rebuilt = conv.zmatrix_to_cartesian(entries)
        self.assertLess(conv.max_distance_deviation([a.xyz for a in atoms], rebuilt), 1e-9)

    def test_zmatrix_round_trip_through_file(self):
        atoms = io.read_pdb(CRAMBIN, backbone_only=True)
        entries = conv.cartesian_to_zmatrix(atoms)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.zmat")
            io.write_zmatrix(entries, path)
            loaded = io.read_zmatrix(path)
        rebuilt = conv.zmatrix_to_cartesian(loaded)
        # The file stores 4 decimals for bonds and 3 for angles.
        self.assertLess(conv.max_distance_deviation([a.xyz for a in atoms], rebuilt), 0.05)

    def test_references_follow_bonds(self):
        atoms = io.read_pdb(CRAMBIN)
        entries = conv.cartesian_to_zmatrix(atoms)
        for i, e in enumerate(entries[1:], start=1):
            self.assertLess(e.bond, 2.1, "atom {} {} bonded to {}".format(i + 1, e.name, entries[e.bond_ref].name))


class ProteinTests(unittest.TestCase):
    def test_crambin_helix_torsions(self):
        torsions = {t.res_seq: t for t in conv.backbone_torsions(io.read_pdb(CRAMBIN))}
        self.assertEqual(len(torsions), 46)
        self.assertIsNone(torsions[1].phi)
        self.assertIsNone(torsions[46].psi)
        # Crambin residues 7-19 form an alpha-helix.
        for seq in range(8, 18):
            self.assertEqual(conv.ramachandran_region(torsions[seq].phi, torsions[seq].psi), "alpha-R")
        for t in torsions.values():
            if t.omega is not None:
                self.assertGreater(abs(t.omega), 160)

    def test_backbone_zmatrix_dihedrals_are_phi_psi_omega(self):
        atoms = io.read_pdb(CRAMBIN, backbone_only=True)
        entries = conv.cartesian_to_zmatrix(atoms)
        torsions = conv.backbone_torsions(atoms)
        for r in range(1, len(torsions)):
            n, ca, c = entries[3 * r], entries[3 * r + 1], entries[3 * r + 2]
            self.assertAlmostEqual(n.dihedral, torsions[r - 1].psi, places=6)
            self.assertAlmostEqual(ca.dihedral, torsions[r - 1].omega, places=6)
            self.assertAlmostEqual(c.dihedral, torsions[r].phi, places=6)

    def test_build_backbone_reproduces_torsions(self):
        rng = random.Random(7)
        n = 12
        phis = [rng.uniform(-180, 180) for _ in range(n)]
        psis = [rng.uniform(-180, 180) for _ in range(n)]
        omegas = [rng.choice([180.0, -175.0, 5.0]) for _ in range(n)]
        atoms = conv.build_backbone(["ALA"] * n, phis, psis, omegas)
        self.assertEqual(len(atoms), 3 * n)
        for i, t in enumerate(conv.backbone_torsions(atoms)):
            if i > 0:
                self.assertAlmostEqual(geo.angle_difference(t.phi, phis[i]), 0.0, places=6)
            if i < n - 1:
                self.assertAlmostEqual(geo.angle_difference(t.psi, psis[i]), 0.0, places=6)
                self.assertAlmostEqual(geo.angle_difference(t.omega, omegas[i]), 0.0, places=6)

    def test_ideal_helix_rise_per_residue(self):
        phi, psi, omega = conv.SECONDARY_STRUCTURE_PRESETS["alpha-helix"]
        n = 20
        atoms = conv.build_backbone(["ALA"] * n, [phi] * n, [psi] * n, [omega] * n)
        ca = [a.xyz for a in atoms if a.name == "CA"]
        # i -> i+3 CA distance in an alpha-helix is about 5.0-5.4 Angstrom.
        self.assertTrue(4.8 < geo.distance(ca[5], ca[8]) < 5.6)


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

    def test_pdb_write_read(self):
        atoms = io.read_pdb(CRAMBIN)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pdb")
            io.write_pdb(atoms, path)
            again = io.read_pdb(path)
        self.assertEqual([a.name for a in atoms], [a.name for a in again])
        for a, b in zip(atoms, again):
            self.assertTrue(all(math.isclose(p, q, abs_tol=1e-3) for p, q in zip(a.xyz, b.xyz)))


if __name__ == "__main__":
    unittest.main()
