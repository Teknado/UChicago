"""Helpers to edit the exam notebook as raw JSON (preserving its exact format) and to
execute a prefix of it with nbclient, writing outputs back into the raw JSON."""
import json, copy, time
import nbformat
from nbclient import NotebookClient

import os
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # the repository root
NB = os.path.join(REPO, 'Final_Autumn_2026-1.ipynb')


def load():
    return json.loads(open(NB, encoding='utf-8').read())


def save(nb):
    with open(NB, 'w', encoding='utf-8') as f:
        f.write(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')


def set_source(nb, idx, text, expect_type):
    cell = nb['cells'][idx]
    assert cell['cell_type'] == expect_type, (idx, cell['cell_type'])
    cell['source'] = text


def execute_prefix(nb, upto, timeout=1800):
    """Execute cells 0..upto (inclusive) in a fresh kernel; copy outputs back. Returns seconds."""
    sub = copy.deepcopy(nb)
    sub['cells'] = sub['cells'][:upto + 1]
    for c in sub['cells']:
        c.pop('id', None)          # nbformat 4.4 has no cell ids
        if isinstance(c.get('source'), list):
            c['source'] = ''.join(c['source'])
    node = nbformat.from_dict(sub)
    client = NotebookClient(node, timeout=timeout, kernel_name='python3',
                            resources={'metadata': {'path': REPO}})
    t = time.perf_counter()
    client.execute()
    secs = time.perf_counter() - t
    for i, c in enumerate(node.cells):
        if c.cell_type == 'code':
            nb['cells'][i]['outputs'] = json.loads(json.dumps(c.outputs))
            nb['cells'][i]['execution_count'] = c.execution_count
    return secs


def text_outputs(nb, idx):
    """Plain-text rendering of a code cell's outputs (for reading results)."""
    out = []
    for o in nb['cells'][idx].get('outputs', []):
        if o['output_type'] == 'stream':
            out.append(o['text'])
        elif o['output_type'] in ('execute_result', 'display_data'):
            d = o['data']
            out.append(d.get('text/plain', '') if 'text/plain' in d else '')
            if 'image/png' in d:
                out.append('[image/png]')
        elif o['output_type'] == 'error':
            out.append('ERROR: ' + o['ename'] + ': ' + o['evalue'])
    return ''.join(x if x.endswith('\n') else x + '\n' for x in out)
