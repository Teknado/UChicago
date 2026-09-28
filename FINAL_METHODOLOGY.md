# Final version: methodology and change record

*BUSN 41210 Financial Analytics, final exam, Autumn 2026. This file documents **`Final_Autumn_2026_final.ipynb`**,
the 41-cell version of the submission, built on 2026-09-28 from `Final_Autumn_2026-1.ipynb` (the 107-cell analysis
notebook, left unchanged). It is self-contained: §1 says what the final file is and how it relates to the analysis
notebook; §2 summarises the methodology and the answers; §3 lists every change made for this version and why; §4
records how it was verified; §5 says how to rebuild it. The full justification of every analytical choice remains
in `METHODOLOGY_LOG.md` (sections C–E and H), `PROJECT_PLAN.md` and `LECTURE_METHODS_P3.md`, which all apply to this version.*

---

## 1. What the final file is

| | |
|---|---|
| File | `Final_Autumn_2026_final.ipynb` |
| Cells | **41** (13 code, 28 markdown), in exactly the order and types of the exam template (commit `174f164`) |
| Built from | `Final_Autumn_2026-1.ipynb` (107 cells, commit `955e6be`), by `notes/final41/build_final41.py` |
| Runs | top to bottom with no errors, about 14 minutes on this machine (4 cores; the Problem 3 cell about 12 of them); data path: the class server's `/classes/41210_MiF_fall2026/Data/`, else the folder of the notebook |
| Numbers | every code statement is the analysis notebook's, so every table, figure and number is the same (verified, §4) |

**Why 41 cells.** The exam does not require a fixed number of cells. Its template's project cell says only
"# Your project starts here." The student chose to keep the template's structure, because a grader's workflow is most
likely built around it. The cost is that each question's code sits in one cell. Problem 3's analysis, 56 cells in the
analysis notebook, becomes one long code cell, and its section notes become appendices. Nothing is lost:
- the code cell prints a banner (`==== Section 3.x: … ====`) where each section starts;
- the notes follow the write-up under the same section numbers, so every "Section 3.x" and "Table 3.x" reference still resolves.

**Cell map** (41-cell slot ← cells of the 107-cell notebook):

| slot | content | from | | slot | content | from |
|---|---|---|---|---|---|---|
| 0–5 | title, AI note, imports, P1 intro, P1 setup, 1.1 question | 0–5 | | 23 | 2.1 answer (the student's, unedited) | 28 |
| 6 | 1.1 code | 6, 7, 8 | | 24 | 2.2 question | 29 |
| 7 | 1.1 answer | 9 | | 25 | 2.2 code | 30, 31 |
| 8 | 1.2 question | 10 | | 26 | 2.2 answer | 32 |
| 9 | 1.2 code (+ the moved 1.2 note, as comments) | 11, 12 | | 27 | 2.3 question | 33 |
| 10 | 1.2 answer | 13 | | 28 | 2.3 code | 34, 35, 36, 37 |
| 11 | 1.3 question | 14 | | 29 | 2.3 answer | 38 |
| 12 | 1.3 code | 15, 16 | | 30 | 2.4 question | 39 |
| 13 | 1.3 answer | 17 | | 31 | 2.4 code | 40, 41, 42 |
| 14 | 1.4 question | 18 | | 32 | 2.4 answer | 43 |
| 15 | 1.4 code | 19, 20 | | 33–34 | P3 intro, data | 44, 45 |
| 16 | 1.4 answer | 21 | | 35 | P3 setup (the exam's code) | 46 |
| 17 | 1.5 question | 22 | | 36–38 | two rules, workflow, what to submit | 47, 48, 49 |
| 18 | 1.5 code (none needed) | 23 | | **39** | **all Problem 3 code, Sections 3.0–3.9** | 51–52, 54–64, 66–68, 70–71, 73–76, 78–88, 90–93, 95–100, 102, 104 |
| 19 | 1.5 answer | 24 | | **40** | **write-up, then appendices: the notes of Sections 3.0–3.10** | 106, then 50, 53, 65, 69, 72, 77, 89, 94, 101, 103, 105 |
| 20–22 | P2 intro, P2 setup, 2.1 question | 25, 26, 27 | | | | |

---

## 2. Methodology and answers (unchanged from the analysis notebook)

The binding rules throughout, set by the student:
- the **AI Coding Guide** is binding: no shuffled split on time-ordered data, and no transformation fitted before the split;
- **scope is the 8 lectures**, and anything not taught is labelled "our construction" or "our choice";
- **every number comes from a cell**;
- **honest reporting**: negative results reported, no benchmark switching, search size stated, anything added after seeing results labelled post hoc.

### Problem 1: trees and ensembles
- The exam's 5-fold stratified CV (`random_state=7034`) throughout. Shuffled folds are legitimate here because the rows are 400 unrelated people.
- Baseline "nobody purchases" 64.25%. Unpruned tree 85.00% (61 leaves, 399/400 in sample).
- CV prefers **3 leaves (90.75%)**, tied with 4 leaves, whose out-of-fold predictions are identical; the tie rule takes the smaller tree.
- The chosen tree splits on Age, then Salary: 97 of 115 people over 42 bought, as did 37 of 44 aged 42 or younger earning over $90,000, but only 9 of 241 of the rest.
- **Forest 88.75% and boosting 89.00%: neither beats the tree**, compared fold by fold on shared folds. The signal is one rectangle, and the untuned ensembles fit noise (99.81% and 97.44% on their training folds).
- MDI ranks Salary first and held-out permutation ranks Age first (25.5 vs 17.1 pp). Salary's greater number of distinct values explains the MDI tilt (L8 p.58). Show the permutation ranking.
- 1.5: shuffled CV on time series leaks the future and biases accuracy upward; use walk-forward evaluation.

### Problem 2: training on your own output (`SEED = 2694`, named random streams)
- **2.1** is the student's own prediction, committed before any Problem 2 code (commit `5086a8b`), and never edited.
- **2.2:** one desk ends at σ̂² = 1.405 × 10⁻⁵ (3.2nd percentile). Across 1,000 desks on night 2,500: median 0.0059, mean 2.34, 57.8% below a tenth of the true VaR, and a mean VaR of 0.775σ against a true 2.326σ. The story's fall is the 52.7th percentile, so it is typical.
- **2.3:** the one-step property holds (one CI misses narrowly, p = 0.0088, which is reported). The nightly drift of log σ̂² is **−1/(n−1)**, which the precise n = 50 run separates from −1/n. The median, the mean and the expectation of 1 are reconciled by the skewed cross-section (the 10 largest desks hold 88.8%).
- **2.4:** keeping the real days holds σ̂² in a band of 0.94–1.06, which is **accumulation of noise**, contained. On dj30, excess kurtosis is 24.1, and the normal model reports a VaR of 2.78% against an empirical 3.33%; losing the tails on night 1 is a **loss of information**. Only real data kept in every generation stops the collapse.

### Problem 3: cross-country return prediction (design fixed in Section 3.0 before any fit)
- **Target** $y = r/\hat\sigma_{t-1}$ (36-month volatility from past returns). Primary benchmark: the pooled trailing mean, updated monthly. Zero, per-class and per-asset means are also reported.
- **Features:**
  - within-class monthly ranks of lagged x1–x5;
  - global macro (log x6, x7–x9) as trailing 60-month z-scores × class;
  - country macro (x86, x12, x13, log x14, log x15, x16) as differentials vs country 7 with trailing z-scores, × class B/C/D. Class A gets no country macro;
  - macro lag 2 months.
- **Data traps:**
  - x10 is a same-year average (look-ahead), so x86 is used instead;
  - x11 stops early for three countries, so it is dropped;
  - eight back-filled series are set to missing, and the first target month is 2003-01;
  - the extended file has duplicates and low-frequency series, so only x86 is taken from it;
  - x5's floor on asset_16 is avoided because σ̂ is computed from returns.
- **Windows:** train 2003–2010; out-of-sample 2011–2024 (168 months, 8,400 asset-months).
- **Schemes and tuning:** static, **expanding** (primary; 14 December refits) and rolling (96 months). Every penalty is tuned inside each training window on 3 forward annual folds.
- **Models:** OLS, **ridge** (primary; unpenalised class intercepts), lasso, PCR, forests (leaf 200; deep forest leaf 5), per-class ridge. 38 pre-listed specifications, with a Bonferroni bar of 3.27 (a 39th post hoc, 3.28).
- **Inference:** a month bootstrap (B = 10,000) on every R²_OOS and difference (our construction), a power analysis, and a block-bootstrap check (SEs about 1.19× larger).
- **Placebo:** every raw macro series is circularly shifted 36–120 months and the pipeline rebuilt.
- **Portfolios:** P1 ∝ ŷ/σ̂ and P3 (within-class long–short) against EW, RP ∝ 1/σ̂ and TSMOM ∝ sign(R12)/σ̂, all at unit gross exposure and rebalanced monthly. Costs are 10 bp per unit of Σ|Δw| (on drifted weights), with a 0–50 bp grid and break-even costs. α regressions on the three rules.
- **Q1: NO.** Ridge M2 R²_OOS = 0.042% (SE 0.241%) against the pooled mean and −0.279% against zero. Characteristics add −0.045% (SE 0.027%) beyond class means. A true R²_OOS of 0.48% would be needed to pass the rule half the time.
- **Q2: NO.** Macro changes R²_OOS by −0.123% (SE 0.085%), and real macro ranks 4 of 9 against the eight placebos.
- **Q3: NO.** P1's net Sharpe is 0.44, against EW 0.28, RP 0.30 and TSMOM 0.12, but α = 0.61% a year with t = 1.74.
  - P1 is essentially the class-means portfolio: weight correlation 0.9994, and α t = 0.82 once that tilt is a regressor.
  - Its edge comes from re-estimating class means each year: frozen at 2010 it earns 0.27.
  - It is concentrated: 60% of gross in class D, asset_16 at 22% on average and 42% at most.

---

## 3. What changed for this version, and why

**No code statement, number, table, figure or verdict changed.** The changes are the layout (§1) and 25 exact text
edits in 12 cells, applied by the build script and shown in full as diffs in `notes/final41/text_changes.md`. They
correct the findings of the comparison review (branch `financial_analytics_review`, `review/COMPARISON.md` §7, found by
blind reviewers and confirmed). Those findings are listed below with what was done about each.

| review finding | change | cells (107-cell numbering) |
|---|---|---|
| **C1** (error). 1.3's second reason, "the forest tries one of the three features at each split, so a split can be forced onto a weak input", is wrong: allowing all three features gives the same 88.75%, and only 6% of the forest's splits are on Gender (checked outside the notebook, so no number from that check is quoted) | The reason is replaced by the correct one, using numbers the notebook prints: the forest's trees are grown until pure (50.2 leaves per tree) and score 99.81% on their training folds; boosting 97.44% against the tree's 91.63% | 17 |
| **C2** (weakness). Answers beyond the exam's limits: 1.2 (one or two sentences), 1.3 (two or three), 2.3 and 2.4 (one paragraph) | 1.2: the two sentences stay; the "technical note" moves into the 1.2 code cell as comments. 1.3: the "why" is three sentences. 2.3: the measurement tables stay, and the reconciliation, including the one with the 2.1 prediction, is one paragraph. 2.4: the two experiments' reports stay, and the discussion is one paragraph. Every number kept is still printed by a cell | 13, 17, 38, 43 |
| **C13** (style). 1.1 quoted "1.25 percentage points of a fold and 0.25 pp of the pooled figure", which no cell prints | Removed | 9 |
| **C4** (weakness). "Pre-registered" in the title is stronger than the notebook can prove | New title: "…an out-of-sample study with a design fixed in advance". The methods section now defines the term ("fixed in Section 3.0 before any model was fitted; the notebook itself cannot prove that order, so every later change is listed in Section 3.10 and every later analysis is labelled post hoc") in place of the version-control sentence | 106 |
| **C3** (weakness). Process jargon | "Development scripts / version control / commit" sentence removed (above); "ledger" defined at first use; "AI guide §4a/§4b" becomes "AI Coding Guide §4a/§4b"; "(unit-tested)" becomes "(checked by a unit test in Section 3.4)". The paper was not shortened (the student chose to meet the exam's limits and remove jargon, not to cut content) | 106 |
| **C5** (weakness). KNN and neural nets were excluded "under the scope rule", as if the exam forbade them | Now: "the exam allows any method, but keeping to methods the lectures teach in full is our choice of scope" | 106 |
| **C6** (style). "Macro lowers R² in every paired comparison" needs "pre-listed" (the post-hoc interaction specification has a small positive point estimate) | All three occurrences now read "every pre-listed paired comparison (Table 3.18)" | 106 |
| Rendering (found while checking this version) | Two unescaped dollar signs in 1.2 ("$90,000 … $90,000") made Jupyter render the text between them as mathematics; they are escaped (`\$90,000`). The same defect is in the 107-cell notebook | 13 |
| Layout | Location words in the moved notes ("the next cell", "checked below", "the harness above", "printed below", "the table below") point to the right places; the write-up's opening note explains the appendices; the deviations list (Section 3.10) gains item 9 describing this version | 50, 72, 77, 89, 103, 105, 106 |

**Deliberately not changed** (with the reason):
- **The student's 2.1 prediction**, which is never edited. 2.3 reconciles against it.
- **The Section 3.0 design rules.** Only two location phrases in its note changed.
- **The word "pre-registered" in the body and in printed tables.** Code outputs use it, so it stays and is now defined.
- **The disclosed approximations:**
  - the month bootstrap treats months as independent (C7);
  - placebo shifts of 72 months or more wrap into early test months (C8);
  - the by-class table uses only the pooled benchmark (C9);
  - the first month's build cost is not charged, and "one-way turnover" appears in the design table (C10).

  Each is already disclosed in the notebook, and fixing it would need new analysis. No analysis was added, following
  the student's rule of proposing new analyses first.
- **Write-up length** (about 6,100 words before the appendices). The student chose not to shorten it.

**On Gemini's claims about the analysis notebook** (verified in `review/GEMINI_FINAL_VERIFICATION.md` on the review
branch): the claims that it names real assets (S&P 500, DAX, Nikkei, Gold…), that it did not formalise the class-means
attribution, and that it emphasised a 2.056σ figure are false. The cell count (107 against the template's 41) was the
one valid structural point. It is not an exam rule, and this version addresses it.

---

## 4. Verification

`notes/final41/verify_final41.py` compares the executed 41-cell notebook with the committed, executed 107-cell notebook:

Result on 2026-09-28 (full output in `notes/final41/verify_output.txt`): **all five checks pass.**

| check | result |
|---|---|
| 1. structure | 41 cells, cell types identical to the exam template. All 22 exam cells are unchanged, except the three setup cells, which add the `_DATA_DIR` fallback; cell 21 also fills in `SEED = 2694`, as the exam asks |
| 2. code | all **2,087** code lines of the 107-cell notebook are in the 41-cell notebook, in order and in the right slot, and there is nothing else (comments and the ten banner prints excepted) |
| 3. text | all **39** markdown cells are present. The 12 corrected cells are shown as diffs in `notes/final41/text_changes.md`; the Problem 3 notes have their headings demoted one level inside the write-up cell |
| 4. outputs | printed text and **all 68 Problem 3 tables** (and every Problem 1–2 table) are identical slot by slot; **20 of 20 figures are byte-identical**. Two differences are explained, not hidden: the Problem 3 runtime in Table 3.31 (a timing), and one count in Table 3.16 (the deep forest with macro, `dB`, "years with a positive importance": 10 → 11). That count's 2011 value is a floating-point zero (4.4 × 10⁻¹⁷ in the saved ledger), so it flips with the last bit of the forest's parallel prediction; the mean importance is identical (0.051%), and the count is quoted nowhere |
| 5. numbers | no number in any answer or in the write-up is newly unprinted. 24 numbers are not found *verbatim* in any output (the 107-cell version had 27); all 24 appear in both versions and are roundings of printed values (for example 0.0059 for a printed 0.00585) |

The notebook was executed once, top to bottom, from a fresh kernel (861 s, 13 of 13 code cells, no errors). Two
markdown-only fixes were made afterwards: the 1.2 dollar signs, and item 9 of Section 3.10 describing them. Because
no code cell changed, the executed outputs were carried over; the script asserts that each code cell's source is identical.

---

## 5. Rebuilding

From the repository root:
```
python3 notes/final41/build_final41.py      # 107-cell notebook -> Final_Autumn_2026_final.ipynb (outputs cleared)
python3 notes/final41/execute.py            # execute it top to bottom (about 14 minutes)
python3 notes/final41/verify_final41.py     # the five checks of §4; also regenerates notes/final41/text_changes.md
```
**Which file to edit.** `Final_Autumn_2026-1.ipynb` remains the analysis notebook and the source for the build.
Either edit it and rebuild, or edit the 41-cell file directly and stop rebuilding; never mix the two, or edits
will be overwritten. Environment: Python 3.11, pandas 3.0.6, scikit-learn 1.9.1, numpy 2.4.6, statsmodels 0.15.0.
The course server has Python 3.14, pandas 3.0.5 and scikit-learn 1.9.0.
