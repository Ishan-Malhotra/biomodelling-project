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

    def test_cartesian_to_internal_with_pdb_example(self):
        text = self.client.get("/api/examples/protein").json()["text"]
        data = self.client.post("/api/cartesian-to-internal", json={"text": text}).json()
        self.assertEqual(len(data["atoms"]), 327)
        self.assertEqual(len(data["torsions"]), 46)
        self.assertEqual(data["check"]["kind"], "distances")
        self.assertLess(data["check"]["max_deviation"], 1e-9)

    def test_typed_lines_and_errors(self):
        ok = self.client.post("/api/cartesian-to-internal", json={"text": "N 0 0 0\nCA 1.458 0 0"})
        self.assertEqual(ok.status_code, 200)
        bad = self.client.post("/api/cartesian-to-internal", json={"text": "N 0 0 0\nCA 1 abc 0"})
        self.assertEqual(bad.status_code, 400)
        self.assertIn("line 2", bad.json()["detail"])

    def test_zmatrix_round_trip(self):
        zmat = self.client.post("/api/cartesian-to-internal",
                                json={"text": self.client.get("/api/examples/cartesian").json()["text"]}).json()
        data = self.client.post("/api/internal-to-cartesian", json={"text": zmat["download"]["text"]}).json()
        self.assertEqual(len(data["atoms"]), 12)
        self.assertTrue(data["download"]["text"].startswith("REMARK"))

    def test_backbone_preset_and_custom(self):
        helix = self.client.post("/api/backbone-to-cartesian", json={"count": 6, "sequence": "AGK", "preset": "alpha-helix"}).json()
        self.assertEqual(len(helix["atoms"]), 18)
        self.assertEqual(helix["atoms"][3]["residue"], "GLY2")
        self.assertLess(helix["check"]["max_error"], 1e-6)

        residues = [{"phi": None, "psi": 120, "omega": 180}, {"phi": -60, "psi": None, "omega": None}]
        custom = self.client.post("/api/backbone-to-cartesian", json={"count": 2, "residues": residues})
        self.assertEqual(custom.status_code, 200)

        missing = self.client.post("/api/backbone-to-cartesian",
                                   json={"count": 2, "residues": [{"psi": 120}, {"phi": -60}]})
        self.assertEqual(missing.status_code, 400)
        self.assertIn("omega", missing.json()["detail"])


if __name__ == "__main__":
    unittest.main()
