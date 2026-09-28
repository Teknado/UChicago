"""Render a notebook as plain text: every cell's source, then its outputs.
HTML tables (pandas Styler / DataFrame displays) are converted to text tables so their numbers are visible.
Usage: python3 review/tools/nb2text.py NOTEBOOK.ipynb OUT.txt"""
import json, sys, io, re, warnings
import pandas as pd
warnings.filterwarnings('ignore')

def html_to_text(html):
    try:
        tabs = pd.read_html(io.StringIO(html))
    except Exception:
        return re.sub(r'<[^>]+>', ' ', html)
    cap = re.search(r'<caption>(.*?)</caption>', html, re.S)
    out = [cap.group(1).strip()] if cap else []
    with pd.option_context('display.max_rows', 500, 'display.max_columns', 60, 'display.width', 250,
                           'display.max_colwidth', 80):
        for t in tabs:
            t = t.loc[:, ~t.columns.astype(str).str.startswith('Unnamed: 0_level')] if False else t
            out.append(t.to_string(index=False))
    return '\n'.join(out)

def outputs(cell):
    r = []
    for o in cell.get('outputs', []):
        t = o.get('output_type')
        if t == 'stream':
            r.append(''.join(o['text']))
        elif t == 'error':
            r.append(f"ERROR {o.get('ename')}: {o.get('evalue')}")
        elif 'data' in o:
            d = o['data']
            if 'text/html' in d:
                r.append(html_to_text(''.join(d['text/html'])))
            elif 'image/png' in d:
                r.append('[FIGURE]')
            elif 'text/plain' in d:
                r.append(''.join(d['text/plain']))
    return '\n'.join(r)

nb = json.load(open(sys.argv[1]))
with open(sys.argv[2], 'w') as f:
    for i, c in enumerate(nb['cells']):
        f.write(f"\n######## CELL {i} [{c['cell_type']}]\n{''.join(c['source'])}\n")
        if c['cell_type'] == 'code':
            f.write('-------- OUTPUT\n' + outputs(c) + '\n')
