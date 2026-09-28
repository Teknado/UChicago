"""Build the submission notebook: the 41-cell notebook with every reference to the project's working process removed.

Run from the repository root:   python3 notes/final_clean/build_clean.py OUTPUT.ipynb
Input : Final_Autumn_2026_final.ipynb (41 cells; unchanged)
Output: OUTPUT.ipynb (outputs cleared; execute it afterwards)

The analysis is untouched. What changes:
- code: proposal codes ("P-A" ... "P-M") and review/audit wording leave the comments, and the eleven post-hoc rows of
  Table 3.31 are labelled "post hoc:" instead of by proposal code (a printed label, not a number);
- markdown: phrases that describe the project's history ("added after review", "after the audit", "this version")
  become neutral post-hoc labels, and the deviations note (Section 3.10) lists only departures from the design.
Every replacement must match exactly the stated number of times.
"""
import json, re, sys, copy

SRC = 'Final_Autumn_2026_final.ipynb'
DST = sys.argv[1]            # e.g. a scratch path; on financial_analytics_final it is Final_Autumn_2026-1.ipynb
nb = json.load(open(SRC))
C = nb['cells']
assert len(C) == 41
txt = [''.join(c['source']) for c in C]

def sub(i, old, new, n=1):
    k = txt[i].count(old)
    assert k == n, (i, k, old[:70])
    txt[i] = txt[i].replace(old, new)

# ---------------------------------------------------------------- code: Problem 1.2 (cell 9)
sub(9, "# Notes on the tree above (moved here from the 1.2 answer, which the exam limits to two sentences):",
       "# Notes on the tree above:")

# Problem 1.2 (cell 9): the fold-stability call collects the fold trees only, but scoring is still set explicitly (the rule
# is that no scorer is ever left at its default).
sub(9, "cv=cv5, return_estimator=True)", "cv=cv5, scoring='accuracy', return_estimator=True)")

# ---------------------------------------------------------------- code: Problem 3 (cell 39)
P = 39
sub(P, "# Post hoc (P-A, added after the final audit at the student's request).", "# Post hoc (added after the out-of-sample results were seen).")
for code in ['P-M', 'P-C', 'P-H', 'P-D', 'P-G']:
    sub(P, f"({code}, post hoc)", "(post hoc)")
sub(P, "(P-E: a post-hoc 39th specification, with its own placebos)", "(post hoc: a 39th specification, with its own placebos)")
sub(P, "(P-B, P-L, post hoc)", "(post hoc)")
sub(P, "(added after the first review)", "(post hoc)")
sub(P, "# ----- the nine post-hoc additions (P-A ... P-M), approved after the final audit", "# ----- post-hoc diagnostics")
k = len(re.findall(r"'P-[A-M]: ", txt[P]))
assert k == 11, k
txt[P] = re.sub(r"'P-[A-M]: ", "'post hoc: ", txt[P])

# ---------------------------------------------------------------- markdown: answers and write-up
for i in range(41):
    if C[i]['cell_type'] == 'markdown' and i not in (1,):
        txt[i] = txt[i].replace('AI guide §', 'AI Coding Guide §')
W = 40
sub(W, "In this version of the notebook, the notes of Sections 3.0–3.10 follow the paper as appendices,",
       "The notes of Sections 3.0–3.10 follow the paper as appendices,")
sub(W, "interactions of the characteristics with macro states, checked after the audit, add nothing either.",
       "interactions of the characteristics with macro states, checked post hoc, add nothing either.")
sub(W, "the diagnostics added after review re-use it and are labelled post hoc.",
       "the post-hoc diagnostics re-use it and are labelled as such.")
sub(W, "were checked after the audit as a labelled 39th specification", "were checked post hoc as a labelled 39th specification")
sub(W, "the regression by half was added after review)", "the regression by half is post hoc)")
sub(W, "which were also outside the list, were run after the audit (Section 4.2)", "which were also outside the list, were run post hoc (Section 4.2)")
sub(W, "added after the first review, as diagnostics.", "post hoc, as diagnostics.")

i0 = txt[W].index('### 3.10 Deviations from the design in Section 3.0')
DEVIATIONS = r"""### 3.10 Deviations from the design in Section 3.0

Every departure from the design of Section 3.0 is listed here, with its effect. The windows, target, benchmarks, primary tests, decision rules, grids, forest settings, placebo shifts and cost are exactly as fixed in Section 3.0.

1. **The leak alarm tests the largest $R^2_{OOS}$.** The design (and L5 p.57) means a suspiciously *high* value, so the alarm is applied to the maximum rather than to the absolute value, which large negative values would set off. It is not triggered: the largest $R^2_{OOS}$ is far below 2%.
2. **The curated-`x10` variant (an extra).** With a 14-month lag, `x10`'s trailing z-score first exists for target 2003-02, so the 50 rows of 2003-01 are set to 0, the neutral value. This affects one training month of one extra specification.
3. **"In-sample fit exceeds out-of-sample fit"** is a sanity expectation, not a guarantee, so it is reported as a count rather than asserted. It holds for all 38 specifications.
4. **Definitions.**
   - Turnover is $\sum_i|\Delta w_i|$, buys plus sells. The design's "one-way turnover" means this sum, not half of it, so the cost is conservative under the half-sum convention.
   - The class risk budget uses the RP sleeve's volatility for P1 too.
   - The forest's leaf of 200 departs from L8 p.44, where a forest's trees are grown deep (our choice; the deep forest tests it).
   - The stored `x11` is left out because of its gaps and its near-redundancy with `x86 − x12`, not because of exact collinearity.
5. **Post-hoc diagnostics.** The analyses below were added after the out-of-sample results were seen. Each is labelled post hoc where it is printed. They re-use the out-of-sample window that the design meant to score once; none changes a pre-registered verdict, and only the interaction model is a specification (the 39th):
   - the average next-month $y$ in each tercile (Table 3.6b); the macro inputs described (Table 3.5b); the spread of the target by class and year (Tables 3.6c–d) and the decile numbers behind Figure 3.6b (Table 3.17b);
   - the power of the actual Q1 statistic (Section 3.5); the per-asset benchmark summary (Table 3.10c); paired scheme differences (Table 3.12b); serial dependence of the monthly losses and a moving-block bootstrap, our construction (Tables 3.12c–d); forest minus ridge, paired (Table 3.12e);
   - $R^2_{OOS}$ at every fixed penalty, an upper bound chosen with hindsight, and the validation curves (Table 3.14c, Figures 3.5c–d); what PCR's components load on (Table 3.14d, Figure 3.5e);
   - the share of country-macro variance that lies across countries (Table 3.18b); where the placebo shift wraps (Table 3.19b); PCR with $K = 0$ allowed (Section 3.6), because the pre-registered PCR grid starts at $K = 1$ and so cannot reach the class-means model that ridge and lasso can reach;
   - characteristic × macro interactions with their own placebos (Table 3.19c), a 39th specification, which moves the Bonferroni value to 3.28;
   - P1 on the static and rolling ridge M2 forecasts and its Sharpe difference from the frozen version (Table 3.29), the regression of P1 on the rules by sub-period (Table 3.27b), the class tilt year by year (Table 3.28b), and diagnostics and a bootstrap SE for the Q3 α regression (Tables 3.27c–e, Figure 3.9).
6. **Order of design and results.** The design of Section 3.0 was written before any model was fitted. The notebook itself cannot prove that order, which is why every later change is listed here and every later analysis is labelled post hoc.
"""
txt[W] = txt[W][:i0] + DEVIATIONS

# ---------------------------------------------------------------- precision: post-hoc labels, wording, rendering
# Captions of analyses added after the out-of-sample results were seen carry "(post hoc)", as Section 3.10 says.
for cap in ["Table 3.6b: average next-month y in each within-class tercile, 2003-2010 (mean over classes and months)",
            "Table 3.10c: against the per-asset trailing mean of y (2011-2024)",
            "Table 3.12b: scheme differences, paired month bootstrap (vs the pooled trailing mean)",
            "Table 3.17b: the numbers behind Figure 3.6b (mean forecast and mean realised y by forecast decile)",
            "Table 3.18b: share of the variance of each country-macro input that lies across countries within a month",
            "Table 3.19b: where the circular shift puts end-of-sample macro values",
            "Table 3.28b: the class tilt year by year (class means re-estimated every December)"]:
    sub(P, cap + "'", cap + " (post hoc)'")
sub(P, "'Table 3.24: portfolios, 2011-2024 (168 months)'", "'Table 3.24: portfolios, 2011-2024 (168 months; the P1 static and rolling rows are post hoc)'")
sub(P, "f'Table 3.25: net Sharpe ratio ({HEADLINE_BP} bp) by sub-period'", "f'Table 3.25: net Sharpe ratio ({HEADLINE_BP} bp) by sub-period (the P1 static and rolling rows are post hoc)'")
sub(P, "'Table 3.26: net Sharpe ratio against the cost per unit of turnover (sum of |change in w|)'",
       "'Table 3.26: net Sharpe ratio against the cost per unit of turnover (sum of |change in w|; the P1 static and rolling rows are post hoc)'")
sub(P, "minus its frozen version, net of 10 bp'", "minus its frozen version, net of 10 bp (the frozen-version row is post hoc)'")
sub(P, "or leverage above 3 times its mean'", "or leverage above 3 times its mean (the 3x cut-off is our choice)'")
sub(P, "f'Table 3.10: {bn}, core specifications, 2011-2024 (8,400 asset-months)'", "f'Table 3.10 (panel: {bn}): core specifications, 2011-2024 (8,400 asset-months)'")
# The design table printed in full (pandas would cut each rule at 50 characters), and the statsmodels version too.
sub(P, "display(design.to_frame())\nimport sys, sklearn\n", "with pd.option_context('display.max_colwidth', None):\n    display(design.to_frame())\nimport sys, sklearn, statsmodels\n")
sub(P, ", scikit-learn {sklearn.__version__}')", ", scikit-learn {sklearn.__version__}, statsmodels {statsmodels.__version__}')")
# Table 3.31: the rows that come from post-hoc analyses are labelled as such.
for lab in ["Q1: true R2_OOS needed to pass the rule at 50% / 80% power", "ridge M2: static minus expanding, paired (SE, t)",
            "paired t, static minus expanding: OLS M2 / RF M2 / PCR M3", "vs the per-asset trailing mean of y: ridge M2 / class means M1 (SE, t)",
            "specifications with R2_OOS > 0 vs the per-asset mean; the per-asset mean vs the pooled mean",
            "country macro: share of variance across countries within a month (classes B-D), min / max",
            "placebo: OOS target months with wrapped macro, s = 72 / 120", "PCR M3 with K = 0 allowed: R2_OOS; refits choosing K = 0",
            "Q3: net Sharpe P1 minus P1 static (bootstrap SE)", "Q3: P1 static: alpha and t on EW + RP + TSMOM",
            "Q3: net Sharpe of P1 static (2010 tilt frozen) / P1 rolling (not selected)", "Q3: P1 rolling turnover / month",
            "Q3: alpha of P1 (t), 2011-2017 / 2018-2024", "Q3: class-D share of P1 by year: first / highest (year) / last",
            "Q3: class-means forecast of y for class B, 2011 / 2024", "Q3: class-means forecast of y for class D, 2011 / highest (year) / 2024"]:
    sub(P, "'" + lab + "'", "'post hoc: " + lab + "'")

# Answer 2.3: the reconciliation with the 2.1 prediction is its own short paragraph, and the formula is approximate.
sub(29, "which equals $-1/n$ for any realistic n.**", "which is approximately $-1/n$ for any realistic n.**")
sub(29, " Against my 2.1 prediction, the Jensen argument", "\n\n**Against my 2.1 prediction.** The Jensen argument")

# Write-up and appendices
sub(W, "Even with the penalty chosen in hindsight, no linear model beats the class means.",
       "Even with the penalty chosen in hindsight (post hoc), no linear model beats the class means by more than about 0.001 percentage points.")
sub(W, "Its weights come from the class means, not the characteristics, and the gain is associated with re-estimating those means each year.",
       "Its weights come from the class means, not the characteristics, and a post-hoc comparison associates the gain with re-estimating those means each year.")
sub(W, "The static fit is worst for every model: on the same months, static minus expanding",
       "The static fit is worst for every model: on the same months (paired differences, post hoc), static minus expanding")
sub(W, "Against each asset's own trailing mean the picture looks better (Table 3.10c):",
       "Against each asset's own trailing mean the picture looks better (post hoc; Table 3.10c):")
sub(W, "**No exploitable nonlinearity** (post hoc;", "**No detectable nonlinearity** (post hoc;")
sub(W, "Lasso keeps no variable in half of its refits.", "No variable is selected in half or more of lasso's refits.")
sub(W, "Frozen at its 2010 estimates, the same model earns", "Frozen at its 2010 estimates (a post-hoc diagnostic), the same model earns")
sub(W, "Frozen at its 2010 estimate, the tilt does worse than risk parity.", "Frozen at its 2010 estimate (a post-hoc diagnostic), the tilt does worse than risk parity.")
sub(W, "the headline test could only detect an R²_OOS of about 0.5%, above the range the literature reports.",
       "the headline test could reliably detect only an R²_OOS of about 0.5–0.7%, at or above the top of the range the literature reports.")
sub(W, "| P1 on ridge M2, static (2010 forecasts frozen; diagnostic) |", "| P1 on ridge M2, static (2010 fit frozen; post hoc) |")
sub(W, "| P1 on ridge M2, rolling (not selected; diagnostic) |", "| P1 on ridge M2, rolling (not selected; post hoc) |")
sub(W, "Months are treated as independent; serial dependence was not checked.",
       "Months are treated as independent; a post-hoc moving-block bootstrap (Tables 3.12c–d, our construction) gives SEs a median of 1.19 times larger.")
sub(W, r"\qquad\text{(L5 p.55–57; never `r2_score`, which uses the test mean: AI Coding Guide §4d)}$$",
       "$$\n\n(L5 p.55–57; never `r2_score`, which uses the test mean: AI Coding Guide §4d.)")
sub(W, "The static version holds the forecasts of 2010 fixed, so it separates",
       "The static version holds the 2010 fit (coefficients and class means) fixed, so it separates")
sub(W, "(L4 p.43; L5 p.41).\n\n### 3.2 Feature engineering",
       "(L4 p.43; L5 p.41). It sorts the volatility-scaled return $y$, the target, rather than the raw return, so that within a class each asset's return counts in units of its own risk.\n\n### 3.2 Feature engineering")
sub(W, "The design's \"one-way turnover\" means this sum, not half of it, so the cost is conservative under the half-sum convention.",
       "The design text says \"one-way turnover\"; the cost is charged on this sum, which is twice one-way turnover under the usual half-sum convention, so the cost is conservative.")

# ---------------------------------------------------------------- no reference to the working process may remain
BANNED = re.compile(r"\bP-[A-S]\b|\breview(?:ed|er|s)?\b|referee|\baudit\b(?! report)|final audit|approv|student's request|"
                    r"\bthe student\b|41-cell|107-cell|this version|version control|\bcommit\b(?! before you compute)|"
                    r"development|Gemini|comparison review|moved here|first run|was corrected|were corrected|tightened", re.I)
ALLOWED_CELLS = {1, 20, 22, 30, 38}          # exam text (the exam's own words, e.g. "Commit before you compute", "preview")
hits = []
for i, t in enumerate(txt):
    if i in ALLOWED_CELLS:
        continue
    for m in BANNED.finditer(t):
        ctx = t[max(0, m.start() - 50):m.end() + 30].replace('\n', ' ')
        if re.search(r'Trap audit|Leakage audit|audit = pd\.Series|display\(audit|audit before interpreting|audited but|audit of data traps|`x10` audit', ctx):
            continue
        hits.append((i, m.group(0), ctx))
assert not hits, hits

out = copy.deepcopy(nb)
for c, t in zip(out['cells'], txt):
    c['source'] = t
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None
json.dump(out, open(DST, 'w'), indent=1, ensure_ascii=False)
open(DST, 'a').write('\n')
print(f'wrote {DST}: 41 cells; no reference to the working process remains outside the exam text')
