"""Read-only diagnosis of native STE versus exact binary forward arithmetic.

An in-memory alternative expression is evaluated on the same frozen weights.
No checkpoint, training method or primary result is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import src.flood_transformer.model as model_module
from revision.models import transformer,phase_switch,predict,seed_all
from revision.data import load_basin
from revision.analysis import binary_overrides
from revision.common import metrics,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-folder',type=Path,default=Path(
        'outputs/steh_population_20261012/01052500/seed_42/L15/steh_with_q'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();folder=args.model_folder
    cfg=json.loads((folder/'run_config.json').read_text());cfg['device']='cuda'
    station=folder.parts[-4];seed_all(cfg['seed']);torch.set_num_threads(1)
    splits,meta=load_basin(cfg['data_root'],station,cfg)
    checkpoint=torch.load(folder/'trained.pt',map_location='cpu',weights_only=False)
    model=transformer(len(meta['feature_names']),cfg)
    model.load_state_dict(checkpoint['state_dict']);model.to('cuda')
    phase_switch(model,'mask',cfg['target_activation_sparsity'])
    base=binary_overrides(model,'cuda')
    native=predict(model,splits['test'],cfg);binary=predict(model,splits['test'],cfg,base)
    probabilities=[torch.sigmoid(p.detach()) for n,p in model.named_parameters() if 'mask_logits' in n]
    errors=torch.cat([(p>=.5).float()+p-p-(p>=.5).float() for p in probabilities])
    original=model_module.ste_binary_mask
    def parenthesized(logits):
        probabilities=torch.sigmoid(logits)
        return (probabilities>=.5).float()+(probabilities-probabilities.detach())
    try:
        model_module.ste_binary_mask=parenthesized
        exact_native=predict(model,splits['test'],cfg)
    finally:model_module.ste_binary_mask=original
    result={
        'station':station,'with_q':cfg['with_q'],
        'checkpoint_sha256':hashlib.sha256((folder/'trained.pt').read_bytes()).hexdigest(),
        'gate_max_roundoff':float(abs(errors).max()),'nonexact_gate_values':int((errors!=0).sum()),
        'gates':len(errors),'max_native_binary_difference_train_scaled':float(abs(native-binary).max()),
        'mean_native_binary_absolute_difference_train_scaled':float(abs(native-binary).mean()),
        'max_parenthesized_native_binary_difference_train_scaled':float(abs(exact_native-binary).max()),
        'native_nse':metrics(native,splits['test']['y'])['nse'],
        'binary_nse':metrics(binary,splits['test']['y'])['nse'],
        'parenthesized_nse':metrics(exact_native,splits['test']['y'])['nse'],
        'exact_expression_bitwise_equal_to_binary':bool(np.array_equal(exact_native,binary)),
        'scope':'Numerical diagnosis at one frozen development model; no training or primary-method change.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(args.output,result);print(json.dumps(result))


if __name__=='__main__':main()
