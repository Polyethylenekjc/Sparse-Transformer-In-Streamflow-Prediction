#!/usr/bin/env python3
import json, csv, statistics, pathlib, sys
from collections import defaultdict

cache_path = pathlib.Path('.streamlit_cache/station_overview__mnt_d_Store_Sparse_Transformer.json')
if not cache_path.exists():
    print('Cache file not found:', cache_path)
    sys.exit(1)

raw = json.load(cache_path.open())
rows = raw.get('rows') if isinstance(raw, dict) and 'rows' in raw else raw
if rows is None:
    print('No rows found in cache')
    sys.exit(1)

items = []
for r in rows:
    nse = r.get('nse')
    nodes = r.get('total_active_nodes')
    # fallback to sum of active parts
    if nodes is None:
        nodes = r.get('active_input_count')
        if nodes is not None:
            nodes = nodes + (r.get('active_head_count') or 0) + (r.get('active_neuron_count') or 0)
    items.append({'station': r.get('station_id'), 'nse': nse, 'nodes': nodes})

groups = [
    ('NSE<0.5', lambda v: v is not None and v < 0.5),
    ('NSE>=0.5', lambda v: v is not None and v >= 0.5),
    ('All', lambda v: True),
]

out = []
for name, cond in groups:
    sel = [it for it in items if cond(it['nse'])]
    node_vals = [it['nodes'] for it in sel if it['nodes'] is not None]
    nse_vals = [it['nse'] for it in sel if it['nse'] is not None]
    out.append({
        'group': name,
        'stations': len(sel),
        'nodes_available': len(node_vals),
        'mean_nodes': round(statistics.mean(node_vals),2) if node_vals else None,
        'median_nodes': round(statistics.median(node_vals),2) if node_vals else None,
        'mean_nse': round(statistics.mean(nse_vals),4) if nse_vals else None,
    })

# Print Markdown table
print('|Group|Stations|Nodes Available|Mean Nodes|Median Nodes|Mean NSE|')
print('|-:|:-:|:-:|:-:|:-:|:-:|')
for r in out:
    print(f"|{r['group']}|{r['stations']}|{r['nodes_available']}|{r['mean_nodes'] or ''}|{r['median_nodes'] or ''}|{r['mean_nse'] or ''}|")

# Save CSV with per-station info
out_csv = pathlib.Path('outputs/aggregate_nodes_by_nse.csv')
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open('w', newline='') as fh:
    writer = csv.writer(fh)
    writer.writerow(['station','nse','nodes'])
    for it in items:
        writer.writerow([it['station'], it['nse'], it['nodes']])

print('\nSaved per-station CSV to', out_csv)
