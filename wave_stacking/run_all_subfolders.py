"""
Run notebook_wave_stacking.ipynb for each SUBFOLDER by swapping the parameter cell.
Usage: conda run -n tpose python wave_stacking/run_all_subfolders.py
"""
import json, subprocess, sys, re, os

SUBFOLDERS = ['3month', '3month_Ri3', '3month_Ri5']
NB         = 'wave_stacking/notebook_wave_stacking.ipynb'

with open(NB) as f:
    nb = json.load(f)

# First code cell is the parameters cell
param_idx = next(i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code')
orig_src  = nb['cells'][param_idx]['source']
orig_src_str = orig_src if isinstance(orig_src, str) else ''.join(orig_src)

for sf in SUBFOLDERS:
    print(f'\n{"="*50}\nRunning SUBFOLDER = {sf}\n{"="*50}')
    new_src = re.sub(r"SUBFOLDER\s*=\s*'[^']*'", f"SUBFOLDER = '{sf}'", orig_src_str)
    nb['cells'][param_idx]['source'] = new_src
    with open(NB, 'w') as f:
        json.dump(nb, f, indent=1)

    result = subprocess.run(
        ['conda', 'run', '-n', 'tpose',
         'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--inplace',
         NB, '--ExecutePreprocessor.timeout=300'],
        capture_output=False)

    if result.returncode != 0:
        print(f'ERROR on {sf} — stopping.')
        sys.exit(1)
    print(f'Done: {sf}')

# Restore original source
nb['cells'][param_idx]['source'] = orig_src_str
with open(NB, 'w') as f:
    json.dump(nb, f, indent=1)
print('\nAll subfolders complete.')
