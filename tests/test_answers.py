import sys, pathlib, unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from selflabel.answers import parse_answer, is_correct

class T(unittest.TestCase):
    def test_answer_line(self):
        self.assertEqual(parse_answer("<think>maybe 5</think>\nSo...\nANSWER: 42"), 42)
        self.assertEqual(parse_answer("<think>x</think>ANSWER: $1{,}024$"), 1024)
        self.assertEqual(parse_answer("<think>x</think>**ANSWER:** -7"), -7)
        self.assertEqual(parse_answer("<think>x</think>ANSWER: 3.0"), 3)
    def test_nan(self):
        self.assertEqual(parse_answer("<think>x</think>ANSWER: NaN"), "NaN")
        self.assertEqual(parse_answer("<think>x</think>ANSWER: No solution exists."), "NaN")
        self.assertTrue(is_correct("NaN", "NaN")); self.assertFalse(is_correct(5, "NaN")); self.assertFalse(is_correct("NaN", 5))
    def test_boxed_fallback(self):
        self.assertEqual(parse_answer("<think>x</think>The answer is \\boxed{116}."), 116)
        self.assertEqual(parse_answer("<think>\\boxed{1}</think>Thus \\boxed{\\dfrac{10}{1}}"), 10)
        self.assertEqual(parse_answer("<think>\\boxed{99}</think>Hence \\boxed{3^2} is wrong, ANSWER: 9"), 9)
        self.assertEqual(parse_answer("<think>\\boxed{99}</think> nothing final"), None)
        self.assertEqual(parse_answer("no think marker ... \\boxed{204}"), 204)
    def test_final_is(self):
        self.assertEqual(parse_answer("<think>x</think>Therefore the final answer is 756."), 756)
    def test_garbage(self):
        self.assertIsNone(parse_answer("<think>x</think>ANSWER: x^2+1"))
        self.assertIsNone(parse_answer(""))
        self.assertFalse(is_correct(None, 5))
    def test_unknown_gold(self):
        self.assertIsNone(is_correct(5, None)); self.assertIsNone(is_correct(None, None))

if __name__ == "__main__":
    unittest.main()
