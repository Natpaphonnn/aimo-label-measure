"""ResidualRecorder must (a) record exactly one state per decode step and skip the prefill, (b) match the teacher-forced states
of the same tokens (what trace_states.py computes offline), on a tiny local model on CPU."""
import pathlib, sys, unittest
import numpy as np, torch
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from features.hooks import ResidualRecorder, case_features, decoder_layers, layer_ids
SMOL = ROOT / "data/models/SmolLM2-135M-Instruct"

@unittest.skipUnless(SMOL.exists(), "local SmolLM2 weights missing")
class Hooks(unittest.TestCase):
    def test_matches_teacher_forcing(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(SMOL); model = AutoModelForCausalLM.from_pretrained(SMOL, dtype=torch.float32).eval()
        n = len(decoder_layers(model)); layers = layer_ids(n); self.assertEqual(len(layers), 3)
        ids = tok(["The sum of 2 and 3 is", "Prime numbers below ten:"], return_tensors="pt", padding=True, padding_side="left")
        rec = ResidualRecorder(model, layers, window=5)
        with rec, torch.inference_mode():
            out = model.generate(**ids, do_sample=False, max_new_tokens=12, min_new_tokens=12)
        f = rec.features(); self.assertEqual(f["n_steps"], 11)   # n - 1 decode steps for n generated tokens
        with torch.inference_mode():
            o = model(input_ids=out, attention_mask=torch.cat([ids["attention_mask"], torch.ones(2, 12, dtype=torch.long)], 1), output_hidden_states=True)
        L = ids["input_ids"].shape[1]
        for l in layers:
            tf = o.hidden_states[l + 1][:, L:, :].float()               # (B, 12, H) teacher-forced states of the generated tokens
            np.testing.assert_allclose(f["answer"][l], tf[:, -2].numpy(), rtol=1e-2, atol=1e-2)          # token that emitted the last one (fp16 storage)
            np.testing.assert_allclose(f["trace"][l], tf[:, -6:-1].mean(1).numpy(), rtol=1e-2, atol=1e-2)
        g = rec.features([12, 7])   # sequence 1 read as if it had stopped after 7 tokens (EOS excluded): states of tokens 1..5
        for l in layers:
            tf = o.hidden_states[l + 1][:, L:, :].float()
            np.testing.assert_allclose(g["answer"][l][1], tf[1, 5].numpy(), rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(g["trace"][l][1], tf[1, 1:6].mean(0).numpy(), rtol=1e-2, atol=1e-2)
            np.testing.assert_allclose(g["answer"][l][0], f["answer"][l][0])
        cf = case_features(f); self.assertIn("dispersion", cf[layers[0]]); self.assertGreaterEqual(cf[layers[0]]["dispersion"], 0.0)

if __name__ == "__main__":
    unittest.main()
