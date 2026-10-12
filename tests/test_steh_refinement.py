import unittest
import numpy as np
import torch
import tempfile
import contextlib
import io
from pathlib import Path
from experiments.refine_steh import backward_reduce, linear_cka, unit_list


class RefinementTests(unittest.TestCase):
    def test_structure_diagnostics_do_not_change_fitted_parameters(self):
        from test_revision import fixture,small_config
        from revision.data import load_basin
        from revision.models import seed_all,transformer,fit
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root)
            cfg=small_config(root)|{'seed':42,'with_q':True,'device':'cpu'}
            splits,meta=load_basin(cfg['data_root'],'01013500',cfg)
            states=[]
            with contextlib.redirect_stdout(io.StringIO()):
                for enabled in [False,True]:
                    seed_all(42);model=transformer(len(meta['feature_names']),cfg)
                    model,logs=fit(model,splits,cfg|{'record_structural_diagnostics':enabled})
                    states.append({k:v.clone() for k,v in model.state_dict().items()})
            self.assertTrue(all(torch.equal(states[0][k],states[1][k]) for k in states[0]))
            self.assertTrue(all('active_gates' in row for row in logs))
            self.assertEqual(sum(row['selected_phase_checkpoint'] for row in logs),3)

    def test_joint_validation_budget_blocks_redundant_pair_removal(self):
        base={'input_mask_override':torch.ones(2),
              'head_mask_overrides':{0:torch.ones(2)},
              'neuron_mask_overrides':{0:torch.ones(8)}}
        def score(mask):
            flags=mask['neuron_mask_overrides'][0]
            # Either redundant neuron alone is sufficient; deleting both is harmful.
            return .8 if flags[0] or flags[1] else -.3
        core,log=backward_reduce(base,score,.05,12)
        self.assertTrue(core['neuron_mask_overrides'][0][:2].sum()>=1)
        self.assertLess(len(unit_list(core)),len(unit_list(base)))
        self.assertTrue(all(r['validation_nse']>=.75 for r in log['trace'] if r['accepted']))

    def test_hidden_comparison_is_invariant_to_unit_permutation_and_scale(self):
        x=np.random.default_rng(42).normal(size=(32,8))
        self.assertAlmostEqual(linear_cka(x,3*x[:,::-1]+5),1.,places=12)
        self.assertIsNone(linear_cka(np.zeros_like(x),x))


if __name__=='__main__':unittest.main()
