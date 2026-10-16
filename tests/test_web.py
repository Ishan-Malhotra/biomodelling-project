import os
import unittest

try:
    from fastapi.testclient import TestClient
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "fastapi is not installed (pip install fastapi httpx)")
class WebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from web.server import app

        cls.client = TestClient(app)

    def test_profile_and_index(self):
        profile = self.client.get("/api/profile").json()
        self.assertIn("name", profile)
        self.assertIn("alpha-helix", profile["presets"])
        self.assertIn("text/html", self.client.get("/").headers["content-type"])

    def test_cartesian_to_internal_from_built_helix(self):
        helix = self.client.post("/api/backbone-to-cartesian",
                                 json={"count": 8, "sequence": "AGKV", "preset": "alpha-helix"}).json()
        text = helix["downloads"][0]["text"]
        data = self.client.post("/api/cartesian-to-internal", json={"text": text}).json()
        self.assertEqual(len(data["atoms"]), 32)
        self.assertEqual(len(data["torsions"]), 8)
        self.assertEqual(data["check"]["kind"], "distances")
        self.assertLess(data["check"]["max_deviation"], 1e-9)

    def test_typed_lines_and_errors(self):
        ok = self.client.post("/api/cartesian-to-internal", json={"text": "N 0 0 0\nCA 1.458 0 0"})
        self.assertEqual(ok.status_code, 200)
        bad = self.client.post("/api/cartesian-to-internal", json={"text": "N 0 0 0\nCA 1 abc 0"})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("line 2", bad.json()["detail"])

    def test_zmatrix_round_trip(self):
        helix = self.client.post("/api/backbone-to-cartesian",
                                 json={"count": 4, "preset": "alpha-helix"}).json()
        zmat = self.client.post("/api/cartesian-to-internal",
                                json={"text": helix["downloads"][0]["text"]}).json()
        self.assertEqual([d["filename"] for d in zmat["downloads"]], ["residues.csv", "zmatrix.zmat"])
        self.assertEqual(len(zmat["torsions"]), 4)
        self.assertIsNotNone(zmat["torsions"][0]["o_dihedral"])
        data = self.client.post("/api/internal-to-cartesian", json={"text": zmat["downloads"][1]["text"]}).json()
        self.assertEqual(len(data["atoms"]), 16)
        self.assertTrue(data["downloads"][0]["text"].startswith("REMARK"))

    def test_backbone_preset_and_custom(self):
        helix = self.client.post("/api/backbone-to-cartesian",
                                 json={"count": 6, "sequence": "AGK", "preset": "alpha-helix"}).json()
        self.assertEqual(len(helix["atoms"]), 24)
        self.assertEqual(helix["atoms"][4]["residue"], "GLY2")
        self.assertEqual(helix["atoms"][3]["name"], "O")
        self.assertLess(helix["check"]["max_error"], 1e-6)

        residues = [{"phi": None, "psi": 120, "omega": 180, "o_angle": 121, "o_dihedral": -55},
                    {"phi": -60, "psi": 140, "omega": None}]
        custom = self.client.post("/api/backbone-to-cartesian", json={"count": 2, "residues": residues}).json()
        self.assertAlmostEqual(custom["torsions"][0]["o_angle"], 121, places=6)
        self.assertAlmostEqual(custom["torsions"][0]["o_dihedral"], -55, places=6)
        self.assertAlmostEqual(custom["torsions"][1]["o_dihedral"], -40, places=6)

        missing = self.client.post("/api/backbone-to-cartesian",
                                   json={"count": 2, "residues": [{"psi": 120}, {"phi": -60, "psi": 0}]})
        self.assertEqual(missing.status_code, 400)
        self.assertIn("omega", missing.json()["detail"])

    def test_residue_csv_builds(self):
        helix = self.client.post("/api/backbone-to-cartesian",
                                 json={"count": 4, "sequence": "AGKV", "preset": "beta-strand"}).json()
        csv_text = helix["downloads"][1]["text"]
        from coordconv import io_utils as io

        residues = [{"res_name": r.res_name, "phi": r.phi, "psi": r.psi, "omega": r.omega,
                     "o_bond": r.o_bond, "o_angle": r.o_angle, "o_dihedral": r.o_dihedral}
                    for r in io.parse_residue_csv(csv_text)]
        data = self.client.post("/api/backbone-to-cartesian", json={"residues": residues}).json()
        self.assertEqual([a["residue"] for a in data["atoms"][::4]], ["ALA1", "GLY2", "LYS3", "VAL4"])
        self.assertLess(data["check"]["max_error"], 1e-6)

    def test_separate_chains_are_refused(self):
        residues = [
            {"res_name": "ALA", "chain": "A", "phi": None, "psi": -47, "omega": 180},
            {"res_name": "ALA", "chain": "B", "phi": -57, "psi": -47, "omega": None},
        ]
        refused = self.client.post("/api/backbone-to-cartesian", json={"residues": residues})
        self.assertEqual(refused.status_code, 400)
        self.assertIn("chain", refused.json()["detail"])

    def test_professor_input_files(self):
        root = os.path.join(os.path.dirname(__file__), "..", "inputs")
        with open(os.path.join(root, "cartesiona_coordinate_Peptide.pdb")) as fh:
            pdb = fh.read()
        peptide = self.client.post("/api/cartesian-to-internal", json={"text": pdb}).json()
        self.assertEqual(len(peptide["atoms"]), 71)
        self.assertEqual([t["residue"] for t in peptide["torsions"]],
                         ["LEU1", "ALA2", "SER3", "ALA4", "ILE5"])
        self.assertLess(peptide["check"]["max_deviation"], 1e-9)

        with open(os.path.join(root, "internal_coordinate.in")) as fh:
            amber = fh.read()
        built = self.client.post("/api/internal-to-cartesian", json={"text": amber}).json()
        self.assertEqual(len(built["atoms"]), 26)
        self.assertEqual(built["atoms"][3]["name"], "N")
        self.assertLess(built["check"]["dihedral"], 1e-6)


if __name__ == "__main__":
    unittest.main()
