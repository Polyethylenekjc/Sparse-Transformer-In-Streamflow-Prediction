"""Global selection must ignore test scores and cover the full tuning panel."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from experiments.global_lstm import choose_global
from test_revision import fixture,small_config

class GlobalLSTMChecks(unittest.TestCase):
    def test_validation_panel_selection(self):
        rows=[{'trial':t,'station':s,'validation_metrics':{'nse':v},'test_nse':100 if t==0 else -100}
              for t,v in [(0,.2),(1,.4)] for s in ['a','b']]
        self.assertEqual(choose_global(rows,['a','b'])[0],1)
        with self.assertRaisesRegex(ValueError,'Incomplete'):choose_global(rows[:-1],['a','b'])

    def test_fresh_single_seed_models_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);fixture(root);cfg=small_config(root)|{'gpu_resident_data':True,'adamw_fused':True,
                  'lstm_trials':[{'hidden_size':32,'num_layers':1,'learning_rate':.001},
                                 {'hidden_size':64,'num_layers':1,'learning_rate':.001},
                                 {'hidden_size':32,'num_layers':2,'learning_rate':.001},
                                 {'hidden_size':64,'num_layers':2,'learning_rate':.0003}]}
            config=root/'config.json';config.write_text(json.dumps(cfg))
            reference=root/'reference.json';reference.write_text(json.dumps({'sample':['01013500'],'population':['01013500']}))
            output=root/'global'
            command=[sys.executable,'experiments/global_lstm.py','--config',str(config),'--reference',str(reference),
                     '--output',str(output),'--trials','4','--tuning-stations','1','--workers','2','--search-seconds','60']
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            final=list(output.glob('final/models/*/summary.json'))
            self.assertEqual(len(final),1)
            result=json.loads(final[0].read_text());decision=json.loads((output/'selected_hyperparameters.json').read_text())
            self.assertEqual(result['seed'],42)
            self.assertEqual(result['hyperparameters'],decision['hyperparameters'])
            self.assertFalse(decision['test_evaluated_for_selection'])
            self.assertEqual(len(list(output.glob('search/trial_*/*/trial_result.json'))),4)
            self.assertNotEqual(str(final[0].parent),str(output/'search'/f"trial_{decision['selected_trial']:03d}"/'01013500'))
            stamp=final[0].read_bytes()
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stdout+run.stderr)
            self.assertEqual(stamp,final[0].read_bytes())

if __name__=='__main__':unittest.main()
