"""Coarse check: every number in the write-up appears somewhere in the executed notebook's outputs
(stream text, text/plain, or the formatted HTML of a Styler table). Prints the numbers that were not found."""
import json, re, html
import os
NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Final_Autumn_2026-1.ipynb')
nb = json.load(open(NB))
cells = nb['cells']
i_p3 = next(i for i, c in enumerate(cells) if c['cell_type'] == 'markdown' and ''.join(c['source']).startswith('## 3.0 Research design'))
out = []
for c in cells[i_p3 - 3:]:
    if c['cell_type'] == 'markdown':
        out.append(''.join(c['source']))      # markdown text of the P3 section (design facts) also counts as a source
        continue
    for o in c.get('outputs', []):
        if o['output_type'] == 'stream':
            out.append(''.join(o['text']))
        elif o['output_type'] in ('execute_result', 'display_data'):
            d = o['data']
            for k in ('text/plain', 'text/html'):
                if k in d:
                    out.append(html.unescape(''.join(d[k])))
blob = '\n'.join(out[:-1])                    # exclude the write-up cell itself
blob_n = blob.replace('−', '-')
wu = ''.join(cells[-1]['source'])
wu = re.sub(r'\$[^$]*\$', ' ', wu)            # skip LaTeX
wu = wu.replace('−', '-')
nums = set(re.findall(r'(?<![\w.])-?\d+(?:\.\d+)?%?', wu))
skip = {str(k) for k in range(0, 13)} | {'2000', '2003', '2010', '2011', '2012', '2013', '2015', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2002', '2008', '2016', '300', '50', '168', '38'}
missing = []
for n in sorted(nums):
    if n in skip or re.fullmatch(r'\d{4}', n):
        continue
    core = n.lstrip('-')
    variants = {n, core}
    if core.endswith('%'):
        v = core[:-1]
        variants |= {v + '%', v + ' %'}
    if not any(v in blob_n for v in variants):
        missing.append(n)
print(f'{len(nums)} distinct numbers in the write-up; not found in any output: {missing}')
