"""Regression checks for batching and independent job scheduling."""
import unittest
import tempfile
import contextlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from src.flood_transformer.sparsity import apply_topk_sparsity_neuronwise
from revision.models import loader, transformer, phase_switch, seed_all, predict


def reference(weight, ratio, axis, minimum):
    if ratio <= 0:
        return weight
    if ratio >= 1:
        return torch.zeros_like(weight)
    keep = max(1, int((1-ratio)*weight.numel()))
    forced = torch.zeros_like(weight, dtype=torch.bool)
    for neuron in range(weight.shape[axis]):
        values = weight[neuron] if axis == 0 else weight[:, neuron]
        if minimum:
            ids = values.abs().topk(min(minimum,len(values)),sorted=False).indices
            if axis == 0:
                forced[neuron,ids] = True
            else:
                forced[ids,neuron] = True
    remaining = max(0, keep-int(forced.sum()))
    scores = weight.abs().flatten().clone()
    scores[forced.flatten()] = -1
    if remaining:
        forced.view(-1)[scores.topk(remaining,sorted=False).indices] = True
    return weight*forced


class PerformanceChecks(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_accelerated_gpu_pipeline(self):
        from test_revision import fixture, small_config
        from revision.runner import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            cfg = small_config(root)|{"device":"cuda","gpu_resident_data":True,"adamw_fused":True}
            path = root/"config.json"
            path.write_text(json.dumps(cfg))
            with contextlib.redirect_stdout(io.StringIO()):
                main(["run","--config",str(path),"--workers","2"])
            summaries = list((root/"results").rglob("summary.json"))
            self.assertEqual(len(summaries),3)
            for p in summaries:
                value = json.loads(p.read_text())
                self.assertTrue(np.isfinite(value["metrics"]["nse"]))
                self.assertTrue(value["training_versions"]["deterministic_algorithms"])
                self.assertEqual(value["training_versions"]["float32_matmul_precision"],"highest")
            self.assertFalse(list((root/"results").rglob("failure.json")))

    def test_parallel_completion_and_resume(self):
        from test_revision import fixture, small_config
        from revision.runner import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            cfg = small_config(root)
            path = root/"config.json"
            path.write_text(json.dumps(cfg))
            with contextlib.redirect_stdout(io.StringIO()):
                main(["run","--config",str(path),"--workers","2"])
            summaries = list((root/"results").rglob("summary.json"))
            self.assertEqual(len(summaries),3)
            original = [p.read_bytes() for p in summaries]
            with contextlib.redirect_stdout(io.StringIO()):
                main(["run","--config",str(path),"--workers","2"])
            self.assertEqual(original,[p.read_bytes() for p in summaries])
            self.assertFalse(list((root/"results").rglob("failure.json")))

    def test_vectorized_projection(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for axis in (0,1):
                for minimum in (0,2,50):
                    for ratio in (0,.2,.9,.999,1):
                        torch.manual_seed(130)
                        weight = torch.randn(7,11,device=device)
                        self.assertTrue(torch.equal(reference(weight,ratio,axis,minimum),
                            apply_topk_sparsity_neuronwise(weight,ratio,axis,minimum)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_resident_sampling_and_prediction(self):
        split = {"x":np.arange(35*4*3,dtype=np.float32).reshape(35,4,3)/100,
                 "y":np.arange(35,dtype=np.float32)}
        cfg = {"batch_size":8,"seed":42,"num_workers":0,"device":"cuda",
               "d_model":8,"n_heads":2,"num_layers":1,"d_ff":8,"dropout":0.}
        cpu_loader = loader(split,cfg,True)
        gpu_loader = loader(split,cfg|{"gpu_resident_data":True},True)
        for epoch in range(3):
            for (x,y),(u,v) in zip(cpu_loader,gpu_loader):
                self.assertTrue(torch.equal(x,u.cpu()))
                self.assertTrue(torch.equal(y,v.cpu()))
        seed_all(42)
        model = transformer(3,cfg).cuda()
        phase_switch(model,"mask",.5)
        np.testing.assert_array_equal(predict(model,split,cfg),
            predict(model,split,cfg|{"gpu_resident_data":True}))


if __name__ == "__main__":
    unittest.main()
