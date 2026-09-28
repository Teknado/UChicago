"""Replace the notebook's Problem 3 cells with the sources in p3/cells_*.py and put the write-up in the last cell."""
import sys
import os as _os
HERE = _os.path.dirname(_os.path.abspath(__file__))              # notes/p3_build
REPO = _os.path.abspath(_os.path.join(HERE, '..', '..'))          # the repository root
sys.path.insert(0, HERE)
import nbtools as T, cells_a, cells_b, cells_c, cells_d, cells_e, cells_f, cells_g, cells_h
nb = T.load()
src = lambda i: ''.join(nb['cells'][i]['source'])
i_setup = next(i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code' and "panel = pd.read_csv(_DATA_DIR + 'asset_panel.csv'" in src(i))
i_design = next(i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'markdown' and src(i).startswith('## 3.0 Research design'))
i_write = len(nb['cells']) - 1
assert nb['cells'][i_write]['cell_type'] == 'markdown' and src(i_write).startswith('## Write-up')
assert src(i_design + 1).startswith('# Your project starts here.\n') and i_setup < i_design < i_write
n_before = len(nb['cells'])
T.set_source(nb, i_setup, cells_a.SETUP_EDIT, 'code')
seq = (cells_a.CELLS_A + cells_b.CELLS_B + cells_c.CELLS_C + cells_d.CELLS_D + cells_e.CELLS_E
       + cells_f.CELLS_F + cells_g.CELLS_G + cells_h.CELLS_H)
def cell(t, s):
    if t == 'md':
        return {'cell_type': 'markdown', 'metadata': {}, 'source': s}
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s}
new = [cell(*seq[0]), cell('code', '# Your project starts here.\n' + seq[1][1])] + [cell(t, s) for t, s in seq[2:]]
nb['cells'][i_design:i_write] = new
nb['cells'][-1]['source'] = open(_os.path.join(HERE, 'writeup.md')).read().rstrip('\n')
T.save(nb)
print(f'P3 cells: {i_write - i_design} -> {len(new)}; notebook cells {n_before} -> {len(nb["cells"])}')
