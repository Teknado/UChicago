"""Build the 41-cell final notebook from the 107-cell analysis notebook.

Run from the repository root:   python3 notes/final41/build_final41.py
Input  : Final_Autumn_2026-1.ipynb   (the 107-cell notebook; never modified)
Output : Final_Autumn_2026_final.ipynb (41 cells in the exam template's order, outputs cleared; execute it afterwards)

What it does, and nothing else:
1. Keeps every exam cell exactly as it is in the 107-cell notebook (they equal the exam template, except the three
   setup cells, which carry the documented `_DATA_DIR` fallback).
2. Merges each answer's code cells into the template's single code cell for that question, in the original order.
   Problem 3's code cells go into the template's one project cell (39), with a printed banner at the start of
   each section 3.0-3.9 so the long output stays navigable.
3. Moves Problem 3's eleven markdown notes (Sections 3.0-3.10) into appendices after the write-up (cell 40).
4. Applies the text corrections listed in EDITS below. Each is an exact replacement that must match once.
No code statement is changed: the only code additions are comments and the section-banner print lines.
"""
import json, os, re, copy

SRC, DST = 'Final_Autumn_2026-1.ipynb', 'Final_Autumn_2026_final.ipynb'
nb = json.load(open(SRC))
C = nb['cells']
assert len(C) == 107, len(C)
src = lambda i: ''.join(C[i]['source'])

# ------------------------------------------------------------------------------------------------------------
# Text corrections: (cell index in the 107-cell notebook, old text, new text). Every `old` must occur exactly once.
# ------------------------------------------------------------------------------------------------------------
EDITS = []

# 1.1: drop derived numbers that no cell prints.
EDITS.append((9, "**How much to read into the gaps.** One person is 1.25 percentage points of a fold and 0.25 pp of the pooled figure. Every size",
                 "**How much to read into the gaps.** Every size"))

# 1.2: the exam asks for one or two sentences; the technical note moves into the code cell as comments (below).
NOTE_12 = src(13)[src(13).index('\n\n*Technical note (outside the two sentences).*'):]
EDITS.append((13, NOTE_12, "\n"))

# 1.2: an unescaped pair of "$90,000" makes Jupyter render the text between them as LaTeX; escape the dollar signs.
EDITS.append((13, "earning more than $90,000 (37 of 44", "earning more than \\$90,000 (37 of 44"))
EDITS.append((13, "people earning $90,000 or less", "people earning \\$90,000 or less"))

# 1.3: the "why" must be two or three sentences, and its second reason was wrong (one feature per split does not
# explain the shortfall: allowing all three features gives the same accuracy). Use the printed training-fold accuracy.
WHY_13_OLD = src(17)[src(17).index('**Why:**'):]
WHY_13_NEW = ("**Why:** With 400 people and effectively two inputs that matter, the buying pattern is close to one rectangle "
              "in the Age × Salary plane (Figure 1.2b), which the 3-leaf tree already draws, so there is little *variance* "
              "left for a forest to average away and little *bias* for boosting to remove (L8 p.51). At their untuned "
              "defaults both ensembles instead fit noise: the forest's trees are grown until their leaves are pure (50.2 "
              "leaves per tree on average) and score 99.81% on their own training folds, and boosting (learning rate 0.1, "
              "depth 3) scores 97.44% there against the small tree's 91.63%. So on data this small and this simple the "
              "readable tree is the better choice, unlike the lecture's much larger housing data, where averaging beats "
              "one tree (L8 p.54).")
EDITS.append((17, WHY_13_OLD, WHY_13_NEW))

# 2.3: the reconciliation must be one paragraph; the measurements stay as tables.
ANS_23_NEW = r"""**Answer:**

**One step from a fixed $\hat\sigma^2_t$ (Table 2.3a).** 200,000 desks were started at the same $\hat\sigma^2_t$ and each took one step of the pipeline:

| start | mean of $\hat\sigma^2_{t+1}/\hat\sigma^2_t$ | 95% CI | contains 1? |
|---|---|---|---|
| $\hat\sigma^2_t = 1$ | 0.99963 | 0.99935 – 0.99991 | no (t = −2.62, p = 0.0088) |
| $\hat\sigma^2_t = 1.405\times10^{-5}$ | 1.00009 | 0.99981 – 1.00036 | yes |

The reporting rule, fixed before running, was "consistent with 1 if the CI contains 1". The second start passes and the first misses narrowly; if the property holds exactly, a miss this large at one of two starts has a 1.7% chance, and the draws were not repeated to make it pass. Unbiasedness of the `ddof=1` variance is exact (L2 p.56), so this is a rare Monte-Carlo miss, and chaining the one-step property makes the expected $\hat\sigma^2$ on night 2,500 exactly 1.

**Average nightly change in $\log\hat\sigma^2$ (Table 2.3b):**

| n | mean nightly change (SE) | n × change (SE) |
|---|---|---|
| 50 | −0.01979 (0.00044) | −0.99 (0.02) |
| 500 | −0.00200 (0.00013) | −1.00 (0.07) |
| 5,000 | −0.000173 (0.000040) | −0.87 (0.20) |
| 50, precise run (2,000 desks × 2,000 nights) | −0.02050 (0.00010) | −1.025 (0.005) |
| 500, the 2.2 run (1,000 desks × 2,499 nights) | −0.002059 (0.000042) | −1.03 (0.02) |

**The formula: the average nightly change in $\log\hat\sigma^2$ is $-1/(n-1)$, which equals $-1/n$ for any realistic n.** n × change is −1 within Monte-Carlo error across two orders of magnitude of n, and the drift is clearly non-zero even at n = 5,000 (t = −4.3). The precise n = 50 run is the only one that can tell the two candidates apart: it is 4.9 SE from −1/n but only 0.9 SE from −1/(n−1). A supplementary, clearly labelled cell checks this against theory beyond the lectures.

**Reconciliation (Figure 2.3, Table 2.3c).** The three numbers are the expectation of $\hat\sigma^2$ on night 2,500, exactly **1**; the median across the 1,000 desks, **0.0059** (a VaR of 0.18σ); and the 1,000-desk mean, **2.34**, of which the 10 largest desks hold **88.8%** and the single largest ($\hat\sigma^2$ = 1,566) 66.8%, while the other 999 average 0.78 and only 5.9% of desks end above 1. Each night multiplies $\hat\sigma^2$ by a random factor whose mean is exactly 1 but whose median is below 1, so on the log scale every desk follows a random walk with a drift of about −1/n per night: after 2,499 nights the median sits near exp(−2,499 × 0.00206) = 0.0058, which matches the measured 0.0059, and the spread grows like $\sqrt{\text{nights}}$ (predicted sd 3.17, observed 3.29). The expectation stays at 1 only because a vanishing minority of desks drifts up enormously and carries the whole average, so a sample of 1,000 desks catches such a desk only now and then and its mean is neither 1 nor the median (here one desk at 1,566 lifts it from 0.78 to 2.34). "Unbiased at every step" does not protect the pipeline, because unbiasedness controls only the *mean* of tomorrow's estimate, not its typical value or its precision (L2 p.71–72): each night's estimation error becomes the next night's truth and is never corrected, since after night 1 no new information about σ enters the library (L2 p.57), so the errors compound multiplicatively and almost every desk's VaR decays toward zero while the expectation, correctly, stays at 1. Against my 2.1 prediction, the Jensen argument and the downward drift of the median at a rate of about −1/(n−1) were confirmed (a mean log ratio of −0.0024 and −0.0019 per night at the two starts, and −0.00206 per night at n = 500 against −1/(n−1) = −0.00200), and so was the severe understatement of VaR (the median desk reports 0.18σ, and 57.8% of desks report less than one-tenth of the truth); the expectation that "the desk average across 1,000 independent banks should remain approximately 1.0" was only half right, because the *expectation* is exactly 1 but the 1,000-desk average is 2.34, and 0.78 without its largest desk: with a distribution this skewed, a sample average of 1,000 desks is not a reliable estimate of the expectation.
"""
EDITS.append((38, src(38), ANS_23_NEW))

# 2.4: the discussion must be one paragraph; the two experiments' reports stay as bullets.
ANS_24_OLD = src(43)[src(43).index('**What synthetic data can and cannot carry.**'):]
ANS_24_NEW = ("**What synthetic data can and cannot carry.** A synthetic sample *contains* exactly the fitted model and nothing "
              "else: a normal distribution with a zero mean and one constant variance, set at the fitted $\\hat\\sigma$ (so its "
              "estimation error is now treated as the truth), plus fresh random noise. It *lacks* any new information about "
              "the world, since it is a function of $\\hat\\sigma$ and a random-number generator (L2 p.57), and everything the "
              "model threw away: the fat tails (kurtosis 24.1 becomes −0.22), the March-2020 cluster, the true 1% quantile "
              "(3.33% rather than 2.78%) and the mean of 0.064% a day, which the model sets to zero. Experiment **(b) shows a "
              "loss of information**: on night 1, before any feedback, the first synthetic sample has already lost the tails, "
              "and they can never come back, because every later generation is drawn from a normal; simulating from the "
              "fitted model \"assumed what the formula assumes\" (L2 p.80). Experiment **(a) shows an accumulation of noise**: "
              "there the model is right (the real days really are normal) and all the information sits in the 500 real days, "
              "so the synthetic half adds only estimation noise, fed back every night, which settles into a small, stable band "
              "(0.94–1.06) because the real half pulls every fit back; remove that anchor, as in 2.2, and the same noise "
              "compounds into a drifting random walk that ends in collapse. For any pipeline retrained on the output of its own "
              "previous version, whether a risk model, a return forecast or a language model, the first generation discards "
              "whatever the model cannot represent, every later generation compounds its predecessor's estimation error, and "
              "being unbiased at every step does not stop the typical run from collapsing (2.3). The only thing that stops it "
              "is **real data, information from outside the model, kept in the training set at every generation** (in (a) even "
              "the same 500 original days are enough), which requires tracking where the data came from; real data stops the "
              "accumulation of noise, but getting the tails back in (b) needs a better model, not more of its own output.")
EDITS.append((43, ANS_24_OLD, ANS_24_NEW))

# Write-up (cell 106)
W = 106
EDITS += [
    (W, "# Characteristics, macro and cross-country asset returns: a pre-registered out-of-sample study",
        "# Characteristics, macro and cross-country asset returns: an out-of-sample study with a design fixed in advance"),
    (W, "because of overlays.*",
        "because of overlays. In this version of the notebook, the notes of Sections 3.0–3.10 follow the paper as appendices, "
        "and the Problem 3 code cell prints a banner at the start of each section.*"),
    (W, "and the list of 38 specifications were written down before any model was fitted (Section 3.0). They were written in "
        "the development scripts first; version control shows them only in a commit made after the development ledger had run "
        "(Section 3.10).",
        "and the list of 38 specifications (the \"ledger\") were written down before any model was fitted (Section 3.0). "
        "Throughout, \"pre-registered\" means fixed in Section 3.0 before any model was fitted; the notebook itself cannot "
        "prove that order, so every later change is listed in Section 3.10 and every later analysis is labelled post hoc."),
    (W, "(AI guide §4a)", "(AI Coding Guide §4a)"),
    (W, "(AI guide §4b)", "(AI Coding Guide §4b)"),
    (W, "(unit-tested)", "(checked by a unit test in Section 3.4)"),
    (W, "- *KNN and neural networks*: the course does not teach KNN regression (Lecture 6 teaches the KNN classifier), and "
        "neural networks are only named, so both are outside the scope rule.",
        "- *KNN and neural networks*: the exam allows any method, but keeping to methods the lectures teach in full is our "
        "choice of scope. The lectures teach the KNN classifier (Lecture 6) but not KNN regression, and neural networks are "
        "only named."),
    (W, "Macro lowers R² in every paired comparison and does no better",
        "Macro lowers R² in every pre-listed paired comparison (Table 3.18) and does no better"),
    (W, "macro loses in every paired comparison and matches its placebos",
        "macro loses in every pre-listed paired comparison (Table 3.18) and matches its placebos"),
    (W, "lowers out-of-sample R² in every paired comparison, macro shifted",
        "lowers out-of-sample R² in every pre-listed paired comparison (Table 3.18), macro shifted"),
]

# Problem 3 notes moved into the appendix: fix the words that referred to neighbouring cells.
EDITS += [
    (50, "This cell and the next were written", "This note and the design constants at the start of the Problem 3 code cell were written"),
    (50, "listed in the next cell (29 core", "listed in the design constants at the start of the Problem 3 code cell (29 core"),
    (72, "checked below.", "checked by the unit tests of the Section 3.4 code."),
    (77, "run through the harness above,", "run through the harness of Section 3.4,"),
    (77, "and is not edited below.", "and is not edited afterwards."),
    (89, "The counts are printed below and checked", "The counts are printed in the Section 3.6 output and checked"),
    (103, "printed in the table below (cell 1:", "printed in Table 3.31, the last table in the output of the Problem 3 code cell (cell 1:"),
]

# The deviations list gains one item describing this version.
EDITS.append((105, "\n\nNothing else changed:",
    "\n9. **This 41-cell version (2026-09-28), after a comparison review.** No code statement, number or result changed. "
    "The code cells of each question were merged into the exam template's single code cell for that question; Problem 3's "
    "code sits in the template's project cell, with a printed banner at the start of each section, and these notes follow "
    "the write-up as appendices. Wording was corrected in seven places: the second reason given in 1.3 (that the forest tries "
    "one feature per split) was replaced, because allowing all three features at every split leaves the forest's accuracy "
    "unchanged, so the answer now cites the printed training-fold accuracy instead; answers 1.2, 1.3, 2.3 and 2.4 were cut "
    "to the exam's sentence and paragraph limits (the technical note of 1.2 moved into its code cell as comments); two "
    "derived numbers that no cell printed were removed from 1.1; the paper's title no longer says \"pre-registered\" and its "
    "methods section says what the term means; the KNN and neural-network exclusion is described as our choice of scope; "
    "\"every paired comparison\" now reads \"every pre-listed paired comparison\"; and two dollar signs in 1.2 are escaped so that the answer renders as text. `FINAL_METHODOLOGY.md` lists every "
    "change.\n\nNothing else changed:"))

text = {i: src(i) for i in range(len(C))}
for i, old, new in EDITS:
    n = text[i].count(old)
    assert n == 1, (i, n, old[:80])
    text[i] = text[i].replace(old, new)

# ------------------------------------------------------------------------------------------------------------
# The 41-cell layout: template slot -> list of 107-cell indices (merged in order)
# ------------------------------------------------------------------------------------------------------------
P3_SECTIONS = [  # (section, title, markdown note, code cells)
    ('3.0', 'Research design, fixed before anything is fitted', 50, [51, 52]),
    ('3.1', 'Know your data', 53, list(range(54, 65))),
    ('3.2', 'Feature engineering', 65, [66, 67, 68]),
    ('3.3', 'Benchmarks first', 69, [70, 71]),
    ('3.4', 'Models and the evaluation harness', 72, [73, 74, 75, 76]),
    ('3.5', 'Running the ledger', 77, list(range(78, 89))),
    ('3.6', 'Does macro add? Placebo and robustness', 89, [90, 91, 92, 93]),
    ('3.7', 'From forecasts to portfolios', 94, list(range(95, 101))),
    ('3.8', 'Leakage audit', 101, [102]),
    ('3.9', 'Headline numbers', 103, [104]),
    ('3.10', 'Deviations from the design in Section 3.0', 105, []),
]
LAYOUT = {0: [0], 1: [1], 2: [2], 3: [3], 4: [4], 5: [5], 6: [6, 7, 8], 7: [9], 8: [10], 9: [11, 12], 10: [13],
          11: [14], 12: [15, 16], 13: [17], 14: [18], 15: [19, 20], 16: [21], 17: [22], 18: [23], 19: [24],
          20: [25], 21: [26], 22: [27], 23: [28], 24: [29], 25: [30, 31], 26: [32], 27: [33], 28: [34, 35, 36, 37],
          29: [38], 30: [39], 31: [40, 41, 42], 32: [43], 33: [44], 34: [45], 35: [46], 36: [47], 37: [48], 38: [49],
          39: 'P3 code', 40: 'write-up + appendix'}

used = sorted(i for v in LAYOUT.values() if isinstance(v, list) for i in v) + \
       sorted([s[2] for s in P3_SECTIONS] + [i for s in P3_SECTIONS for i in s[3]] + [106])
assert used == list(range(107)), 'every one of the 107 cells must be used exactly once'

BANNER = "print('\\n==== Section {s}: {t} ====')"          # printed marker; the verifier strips these lines

def banner(s, t):
    rule = '# ' + '=' * 110
    return f"{rule}\n# Section {s}: {t}   (notes: Appendix, Section {s}, after the write-up)\n{rule}\n" + BANNER.format(s=s, t=t)

def code_cell(source):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source}

def md_cell(source):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source}

new = []
for slot in range(41):
    spec = LAYOUT[slot]
    if spec == 'P3 code':
        first = text[51]
        assert first.startswith('# Your project starts here.\n')
        parts = ['# Your project starts here.\n'
                 '# All of Problem 3 runs in this one cell, in the order of Sections 3.0-3.9. Each section starts with a\n'
                 '# printed banner; its notes are in the appendices after the write-up (next cell).']
        for s, t, _, codes in P3_SECTIONS:
            if not codes:
                continue
            body = [text[i] for i in codes]
            if codes[0] == 51:
                body[0] = body[0][len('# Your project starts here.\n'):]
            parts.append(banner(s, t) + '\n\n' + '\n\n'.join(b.rstrip('\n') for b in body))
        new.append(code_cell('\n\n'.join(parts) + '\n'))
    elif spec == 'write-up + appendix':
        notes = []
        for s, t, md, _ in P3_SECTIONS:
            note = re.sub(r'^(#{2,5}) ', lambda m: '#' + m.group(1) + ' ', text[md], flags=re.M)   # demote headings one level
            notes.append(note.rstrip('\n'))
        appendix = ('\n\n---\n\n## Appendix: the notes of Sections 3.0–3.10\n\n'
                    '*These notes accompany the Problem 3 code cell above, section by section (its output prints a banner '
                    'where each section starts). Section 3.0 is the design written before any model was fitted; Section '
                    '3.10 lists every later deviation.*\n\n' + '\n\n'.join(notes) + '\n')
        new.append(md_cell(text[106].rstrip('\n') + appendix))
    else:
        cells = [C[i] for i in spec]
        kinds = {c['cell_type'] for c in cells}
        assert len(kinds) == 1, (slot, kinds)
        if kinds == {'markdown'}:
            assert len(spec) == 1
            new.append(md_cell(text[spec[0]]))
        else:
            body = text[spec[0]] if len(spec) == 1 else '\n\n'.join(text[i].rstrip('\n') for i in spec)   # single cells verbatim
            if slot == 9:                                   # the 1.2 technical note, kept as comments next to the tree code
                note = NOTE_12.replace('*Technical note (outside the two sentences).*', 'Notes on the tree above (moved here '
                       'from the 1.2 answer, which the exam limits to two sentences):').strip('\n')
                body += '\n\n' + '\n'.join('# ' + l if l.strip() else '#' for l in note.splitlines())
            new.append(code_cell(body if len(spec) == 1 else body + '\n'))

out = copy.deepcopy(nb)
out['cells'] = new
assert len(new) == 41
json.dump(out, open(DST, 'w'), indent=1, ensure_ascii=False)
open(DST, 'a').write('\n')
print(f'wrote {DST}: {len(new)} cells ({sum(c["cell_type"] == "code" for c in new)} code), {len(EDITS)} text edits applied')
