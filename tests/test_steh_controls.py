"""Structural-control regressions for the no-alternative nearest-degree case."""
import unittest
import numpy as np
from experiments.steh_control_sensitivity import draw_group


class CaliperControlsTest(unittest.TestCase):
    def test_caliper_supplies_comparable_alternatives_when_nearest_is_self(self):
        pool=list(range(9));degrees=[[i,2*i] for i in pool]
        nearest=[];caliper=[]
        for seed in range(40):
            nearest.extend(draw_group(pool,[4],degrees,np.random.default_rng(seed),0)[0])
            selected,distance=draw_group(pool,[4],degrees,np.random.default_rng(seed),1)
            self.assertLessEqual(distance[0],1)
            caliper.extend(selected)
        self.assertEqual(set(nearest),{4})
        self.assertEqual(set(caliper),{3,4,5})

    def test_counts_and_sampling_without_replacement_are_preserved(self):
        pool=list(range(9));degrees=[[i,2*i] for i in pool]
        a=draw_group(pool,[3,4,5],degrees,np.random.default_rng(42),2)
        b=draw_group(pool,[3,4,5],degrees,np.random.default_rng(42),2)
        self.assertEqual(a,b)
        self.assertEqual(len(a[0]),3)
        self.assertEqual(len(set(a[0])),3)
        self.assertTrue(set(a[0]).issubset(pool))


if __name__=='__main__':unittest.main()
