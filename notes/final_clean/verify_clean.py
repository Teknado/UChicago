"""Verify the executed submission notebook against the executed 41-cell notebook, and against the course's rules.

Run from the repository root:  python3 notes/final_clean/verify_clean.py SUBMISSION.ipynb REFERENCE_41.ipynb [METHODOLOGY.md]
Checks:
  1. structure: 41 cells with the exam template's cell types; exam cells unchanged (setup cells: data path and SEED);
  2. code: the same statements as the reference, except the eleven relabelled rows of Table 3.31;
  3. outputs: identical to the reference once those labels are mapped (timings, and one floating-point count, explained);
  4. numbers: no number in an answer or the write-up that no cell prints, beyond those the reference already had;
  5. AI Coding Guide: a static scan of every code line for the guide's traps;
  6. no reference to the working process in code, text or outputs (the exam's own text excepted);
  7. (optional) every number in the methodology document appears in the notebook (outputs, code or text).
"""
import json, re, html, sys, subprocess, difflib, hashlib

SUB, REF = sys.argv[1], sys.argv[2]
METH = sys.argv[3] if len(sys.argv) > 3 else None
new, old = json.load(open(SUB))['cells'], json.load(open(REF))['cells']
tmpl = json.loads(subprocess.run(['git', 'show', '174f164:Final_Autumn_2026-1.ipynb'], capture_output=True, text=True, check=True).stdout)['cells']
src = lambda c: ''.join(c['source'])
ok = True
def report(name, problems):
    global ok
    ok &= not problems
    print(f'[{"PASS" if not problems else "FAIL"}] {name}' + ('' if not problems else ':\n  - ' + '\n  - '.join(map(str, problems))))
RELABEL = lambda t: re.sub(r"'P-[A-M]: ", "'post hoc: ", t)
RELABEL_OUT = lambda t: re.sub(r'\bP-[A-M]: ', 'post hoc: ', t)

# 1 ---------------------------------------------------------------- structure
p = []
if len(new) != 41 or [c['cell_type'] for c in new] != [c['cell_type'] for c in tmpl]:
    p.append('cell count or types differ from the template')
free = {s for s, t in enumerate(tmpl) if src(t).strip() in ('**Answer:**', '# Your codes here.', '# Your codes here (if any).', '# Your project starts here.', '## Write-up')}
for s, t in enumerate(tmpl):
    if s in free or src(new[s]) == src(t):
        continue
    d = [l for l in difflib.ndiff(src(t).splitlines(), src(new[s]).splitlines()) if l[:2] in ('+ ', '- ')]
    if s in (4, 21, 35) and all('_DATA_DIR' in l or 'import os' in l or (s == 21 and 'SEED' in l) or not l[2:].strip() for l in d):
        continue
    p.append(f'exam cell {s} changed: {d[:3]}')
report('structure: 41 cells, template cell types, exam cells unchanged (setup cells: data path; cell 21 fills in SEED)', p)

# 2 ---------------------------------------------------------------- code
def statements(code):
    return [l.rstrip() for l in code.splitlines() if l.strip() and not l.strip().startswith('#')]
p = []
for s in range(41):
    if new[s]['cell_type'] == 'code':
        a = statements(RELABEL(src(old[s])).replace("cv=cv5, return_estimator=True)", "cv=cv5, scoring='accuracy', return_estimator=True)"))
        b = statements(src(new[s]))
        if a != b:
            p.append(f'cell {s}: ' + str([l for l in difflib.unified_diff(a, b, lineterm='', n=0)][:6]))
n = sum(len(statements(src(c))) for c in new if c['cell_type'] == 'code')
report(f'code: all {n} code lines equal the reference (only the eleven Table 3.31 labels and one explicit scoring= changed)', p)

# 3 ---------------------------------------------------------------- outputs
TIMING = re.compile(r'(\d+(?:\.\d+)?) ?(s|min|seconds)\b')
def outs(c):
    text, imgs = [], []
    for o in c.get('outputs', []):
        if o['output_type'] == 'stream':
            text.append(''.join(o['text']))
        elif o['output_type'] in ('display_data', 'execute_result'):
            d = o['data']
            if 'image/png' in d:
                imgs.append(hashlib.sha1(''.join(d['image/png']).encode()).hexdigest())
            if 'text/html' in d:
                text.append(re.sub(r'T_[0-9a-f]{5}', 'T_id', ''.join(d['text/html'])))
            elif 'text/plain' in d and 'image/png' not in d:
                text.append(''.join(d['text/plain']))
        elif o['output_type'] == 'error':
            text.append('ERROR')
    return TIMING.sub('<t>', ''.join(text)), imgs
def drop_explained(t):
    t = re.sub(r'Problem 3 runtime \(minutes\).*?</tr>', '', t, flags=re.S)
    t = re.sub(r'(Table 3\.16: rf\|M3\|expanding.*?<td id="T_id_row3_col1" class="data row3 col1" >)1[01](</td>)', r'\1N\2', t, flags=re.S)
    return t
p, same, total = [], 0, 0
for s in range(41):
    if new[s]['cell_type'] != 'code':
        continue
    (ta, ia), (tb, ib) = outs(old[s]), outs(new[s])
    ta, tb = drop_explained(RELABEL_OUT(ta)), drop_explained(tb)
    if ta != tb:
        d = [l for l in difflib.unified_diff(ta.splitlines(), tb.splitlines(), lineterm='', n=0) if not l.startswith(('---', '+++', '@@'))]
        p.append(f'cell {s}: {d[:4]}')
    if len(ia) != len(ib):
        p.append(f'cell {s}: {len(ia)} figures in the reference, {len(ib)} here')
    total += len(ib); same += sum(x == y for x, y in zip(ia, ib))
report(f'outputs: identical to the reference (labels mapped; timings and the Table 3.16 floating-point count excepted); '
       f'{same} of {total} figures byte-identical', p)

# 4 ---------------------------------------------------------------- numbers in answers and write-up
def printed(cells):
    o = []
    for c in cells:
        for x in c.get('outputs', []):
            if x['output_type'] == 'stream':
                o.append(''.join(x['text']))
            elif x['output_type'] in ('execute_result', 'display_data'):
                for k in ('text/plain', 'text/html'):
                    if k in x['data']:
                        o.append(html.unescape(''.join(x['data'][k])))
    return '\n'.join(o).replace('−', '-')
SKIP = {str(k) for k in range(0, 13)} | {'300', '50', '168', '38', '400', '500', '1,000', '2,500', '2,499'}
def nums(t):
    t = re.sub(r'\$[^$]*\$', ' ', t).replace('−', '-')
    return {x for x in re.findall(r'(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?', t) if x not in SKIP and not re.fullmatch(r'\d{4}', x)}
_cache = {}
def found(x, blob):
    core = x.lstrip('-')
    if any(v in blob for v in {x, core, core.replace(',', ''), core.rstrip('%'), core.rstrip('%') + ' %'}):
        return True
    if id(blob) not in _cache:        # every number printed, as floats (a rounding of one of them also counts)
        _cache[id(blob)] = {abs(float(v.replace(',', ''))) for v in re.findall(r'\d[\d,]*\.?\d*(?:e-?\d+)?', blob) if v.replace(',', '').replace('.', '', 1).replace('e-', '', 1).replace('e', '', 1).isdigit()}
    val = core.rstrip('%').replace(',', '')
    try:
        target, dec = float(val), (len(val.split('.')[1]) if '.' in val else 0)
    except ValueError:
        return False
    scale = [1, 100, 0.01] if core.endswith('%') else [1]
    return any(round(p * k, dec) == target for p in _cache[id(blob)] for k in scale)
def missing(cells):
    blob = printed(cells) + '\n'.join(src(c) for c in cells if c['cell_type'] == 'markdown' and not src(c).lstrip().startswith(('**Answer', '## Write-up')))
    return {x for c in cells if c['cell_type'] == 'markdown' and src(c).lstrip().startswith(('**Answer', '## Write-up')) for x in nums(src(c)) if not found(x, blob)}
m_new, m_old = missing(new), missing(old)
report(f'numbers: {len(m_new)} numbers in answers/write-up not found verbatim in any output (reference: {len(m_old)}); new: {sorted(m_new - m_old)}', sorted(m_new - m_old))

# 5 ---------------------------------------------------------------- AI Coding Guide static scan
import io, tokenize
def no_comments(t):
    """The code with every comment token removed (strings such as '#2a78d6' are kept)."""
    toks = [tk for tk in tokenize.generate_tokens(io.StringIO(t).readline) if tk.type != tokenize.COMMENT]
    return tokenize.untokenize(toks)
code = {s: no_comments(src(c)) for s, c in enumerate(new) if c['cell_type'] == 'code'}
P3 = code[39]
rules = [
    ('§4a shuffle=True only in the exam\'s Problem 1 folds', lambda: [s for s, t in code.items() if 'shuffle=True' in t and s != 4]),
    ('§4a train_test_split only in Problem 1.4 (a cross-section)', lambda: [s for s, t in code.items() if 'train_test_split(' in t and s != 15]),
    ('§4a no KFold / GridSearchCV / *CV estimator / cross_val_* in Problem 3',
     lambda: re.findall(r'\b(?:KFold|StratifiedKFold|GridSearchCV|RandomizedSearchCV|RidgeCV|LassoCV|ElasticNetCV|cross_val_score|cross_validate|cross_val_predict|TimeSeriesSplit)\b', P3)),
    ('§4b every StandardScaler / PCA fit happens inside a function that receives the training rows',
     lambda: [l.strip() for l in P3.splitlines() if re.search(r'(StandardScaler|PCA)\(', l) and not l.startswith((' ', '\t', '#')) and 'import' not in l]),
    ('§4c no LogisticRegression', lambda: [s for s, t in code.items() if 'LogisticRegression' in t]),
    ('§4d no r2_score', lambda: [s for s, t in code.items() if 'r2_score' in t]),
    ('scoring= set explicitly in every cross_val_score / cross_validate / permutation_importance',
     lambda: [m.group(0)[:60] for s, t in code.items() for m in re.finditer(r'(cross_val_score|cross_validate|permutation_importance)\((?:[^()]|\([^()]*\))*\)', t) if 'scoring=' not in m.group(0)]),
    ('no n_iter_no_change (shuffled internal split)', lambda: [s for s, t in code.items() if 'n_iter_no_change' in t]),
    ('RandomForestRegressor always given max_features', lambda: [m.group(0)[:60] for m in re.finditer(r'RandomForestRegressor\((?:[^()]|\([^()]*\))*\)', P3) if 'max_features' not in m.group(0) and not ('**params' in m.group(0) and re.search(r'RF_PARAMS = dict\([^)]*max_features', P3))]),
    ('§3 no pandas-1 idioms or sns.set() (DataFrame.append would raise in pandas 3; the run has no errors)', lambda: [s for s, t in code.items() if re.search(r"(?:df|frame|DataFrame)\w*\.append\(|fillna\(method|iteritems\(|normalize=True|sns\.set\(", t)] + [s for s, c in enumerate(new) for o in c.get('outputs', []) if o.get('output_type') == 'error']),
    ('ddof never left to numpy\'s default in .var()/.std() on arrays', lambda: [l.strip()[:70] for s, t in code.items() for l in t.splitlines() if re.search(r'np\.(var|std)\((?![^)]*ddof)', l)]),
]
for name, fn in rules:
    r = fn()
    report(f'AI Coding Guide {name}', [str(x) for x in r])

# 6 ---------------------------------------------------------------- no reference to the working process
BANNED = re.compile(r"\bP-[A-S]\b|\breview(?:ed|er|s)?\b|referee|final audit|approv|student's request|\bthe student\b|41-cell|107-cell|"
                    r"this version|version control|\bcommit\b(?! before you compute)|development|Gemini|comparison review|moved here|first run|"
                    r"was corrected|were corrected|tightened", re.I)
EXAM = {1, 20, 22, 30, 38}
hits = []
for s, c in enumerate(new):
    parts = [src(c)] if s not in EXAM else []
    if c['cell_type'] == 'code':
        parts.append(printed([c]))
    for t in parts:
        hits += [(s, m.group(0), t[max(0, m.start() - 40):m.end() + 30].replace('\n', ' ')) for m in BANNED.finditer(t)]
report('no reference to the working process (code, text and printed output; the exam\'s own text excepted)', hits)

# 7 ---------------------------------------------------------------- methodology numbers
if METH:
    blob = printed(new) + '\n'.join(src(c) for c in new)
    blob = blob.replace('−', '-')
    mt = open(METH).read()
    mt = re.sub(r'`[^`]*`', ' ', mt)
    mnums = {x for x in re.findall(r'(?<![\w.^])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.^,])-?\d+(?:\.\d+)?%?', mt.replace('−', '-')) if x not in SKIP and not re.fullmatch(r'\d{4}', x)}
    miss = sorted(x for x in mnums if not found(x, blob))
    report(f'methodology: {len(mnums)} distinct numbers; all appear in the notebook (outputs, code or text)', miss)

print('\nALL CHECKS PASSED' if ok else '\nSOME CHECKS FAILED')
sys.exit(0 if ok else 1)
