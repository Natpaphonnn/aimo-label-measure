import sys, pathlib, unittest, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from selflabel.perturb import make_variant, MATH_SPAN

P1 = "Let $ABC$ be a triangle with $AB = 20$ and $AC = 26$. Point $D$ lies on segment $BC$ such that $\\angle BAD = \\angle CAD$. Find $BD$."
P2 = "Alice and Bob each hold some integer number of sweets. Alice says to Bob: ``If we each added our ages, my answer would be double yours.'' How many sweets does Alice hold?"
P3 = "Find the number of positive integers $n$ less than $1000$ such that $n^2 + 1$ is divisible by $13$."
P4 = "A circle with center $O$ passes through points $A$ and $B$. The segment from A to B has length $10$. A second circle is tangent to line AB at B. Find the radius."

def inverse_roundtrip(orig, out, note):
    lm = {}
    for part in note.split("; "):
        if part.startswith("labels "):
            for ab in part[7:].split(","):
                a, b = ab.split(">"); lm[b] = a
    back = re.sub(r"[A-Z]", lambda m: lm.get(m.group(0), m.group(0)), out)
    return back

class T(unittest.TestCase):
    def test_rename_labels_consistent_roundtrip(self):
        for seed in range(6):
            out, note = make_variant(P1, "rename", seed)
            self.assertIsNotNone(note); self.assertNotEqual(out, P1)
            self.assertEqual(inverse_roundtrip(P1, out, note), P1, (out, note))
    def test_rename_prose_article_safe(self):
        for seed in range(6):
            out, note = make_variant(P4, "rename", seed)
            self.assertTrue(out.startswith("A circle"), out); self.assertIn("A second circle", out)
            self.assertIn("$O$", out)  # O is never renamed (skipped letter)
            self.assertEqual(inverse_roundtrip(P4, out, note), P4, (out, note))
    def test_rename_names(self):
        out, note = make_variant(P2, "rename", 2)
        self.assertNotIn("Alice", out); self.assertNotIn("Bob", out); self.assertIn("names", note)
    def test_rename_nothing(self):
        out, note = make_variant(P3, "rename", 3); self.assertEqual(out, P3); self.assertIsNone(note)
    def test_typos_keep_math_and_numbers(self):
        for seed in range(5):
            out, note = make_variant(P1, "typos", seed)
            self.assertEqual(MATH_SPAN.findall(out), MATH_SPAN.findall(P1)); self.assertEqual(re.findall(r"\d+", out), re.findall(r"\d+", P1))
    def test_distract(self):
        out, note = make_variant(P3, "distract", 0)
        self.assertGreater(len(out), len(P3)); self.assertEqual(MATH_SPAN.findall(out)[: len(MATH_SPAN.findall(P3))], MATH_SPAN.findall(P3))

if __name__ == "__main__":
    unittest.main()
