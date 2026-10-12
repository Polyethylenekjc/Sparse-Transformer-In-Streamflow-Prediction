import unittest
import numpy as np
import torch
from experiments.refine_steh import backward_reduce, linear_cka, unit_list


class RefinementTests(unittest.TestCase):
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
