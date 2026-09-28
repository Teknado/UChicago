# Context for Claude: BUSN 41210 Financial Analytics final project (Chicago Booth, Autumn 2026)

Read this first. It records what the project is, the rules the student set, what has been done, where everything is,
and how to change things safely. The detailed records are `METHODOLOGY_LOG.md` (every decision and its justification),
`PROJECT_PLAN.md` (the approved plan), and `LECTURE_METHODS_P3.md` (every lecture method and how it relates to Problem 3).

## 1. What this is

- A final exam for BUSN 41210 Financial Analytics (lectures by Dacheng Xiu). It is an **open-AI assignment**: the
  professor explicitly permits AI help. All work is academic.
- The deliverable is **one Jupyter notebook**, `Final_Autumn_2026-1.ipynb`, filled in place. It holds three problems:
  - **Problem 1** (20 points): trees and ensembles.
  - **Problem 2** (20 points): training on your own output.
  - **Problem 3** (60 points): an open-ended research project on cross-country asset return prediction, with a
    research-paper write-up in the notebook's final markdown cell.
- **Status: complete.** All three problems are answered. The notebook runs top to bottom in about 11 minutes with no
  errors. It has been through an independent referee review, a 12-reviewer audit against the AI Coding Guide and all
  8 lectures, and nine approved post-hoc additions.

## 2. The student's standing rules (always follow them)

1. **The AI Coding Guide (`AI_Coding_Guide.pdf`) is binding.** Two errors are fatal:
   - a shuffled split or fold on time-ordered data;
   - a scaler, imputer, PCA or any other transformation fitted before the split.

   Also: `LogisticRegression()` is L2-penalised by default, and `r2_score` uses the test-set mean, so it is **not**
   the course's R²_OOS.
2. **Scope = the 8 lectures** (`Lecture_1.pdf` … `Lecture_8.pdf`). Every model, formula and approach must be grounded
   in them.
   - A method a lecture only *names* counts as **not taught**. Examples: neural nets, SVM, elastic net, HAC SEs,
     t-SNE, Benjamini–Hochberg.
   - Anything not taught is labelled "our construction", "our choice" or "[EXAM-DEFINED]".
3. **Every number must come from a cell** (exam cell 1). Every number in an answer or in the write-up must be printed
   by a code cell. Problem 3's cell "Table 3.31: the numbers the write-up quotes" collects the headline numbers.
4. **Plan first, then ask; do not code until the student approves.** The student likes to:
   - review a plan (with lecture/slide citations) before any coding;
   - be asked clarifying questions before work starts;
   - approve new analyses before they are added.

   Agreed rule from the final audit: **fix outright errors directly; propose any new model or analysis and wait for
   approval.**
5. **Keep the methodology log current.** Every change of plan or result gets a dated row in `METHODOLOGY_LOG.md` §H,
   and Problem 3's changes also go in the notebook's §3.10 "Deviations" cell.
6. **Honest reporting.**
   - Report results whichever way they come out; a negative result is a result.
   - Never switch a benchmark after seeing results. Put the search size next to every headline.
   - Anything added after results were seen is labelled **post hoc**.
7. When the student forwards another AI's critique, evaluate it critically. Accept each point with a reason, or rebut
   it with structured evidence.

## 3. Decisions the student made (do not reopen them without asking)

- `SEED = 2694` for Problem 2 (the exam's student-ID seed). Problems 1 and 3 use 7034 (`random_state=7034`, `P3_SEED`).
- The Problem 2.1 prediction was written by the student and committed verbatim *before* any Problem 2 code (commit
  `5086a8b`). Never edit it.
- Problem 3 scope: pooled models with class-specific terms, plus PCR and per-class models.
- Primary forecast benchmark: the **pooled trailing mean** of the target. Zero, per-class and per-asset means are
  always reported too.
- Costs: **10 bp** per unit of turnover Σ|Δw| as the headline, a 0–50 bp grid, and break-even costs.
- Literature to cite: Gu, Kelly & Xiu (2020); Asness, Moskowitz & Pedersen (2013), which covers value and momentum
  only; Moskowitz, Ooi & Pedersen (2012); the course's 0.3–0.5% monthly R²_OOS sanity range (L5 p.57). **The student
  verifies the references.**
- The notebook is filled **in place** (no copy).

## 4. Repository map

| file | what it is |
|---|---|
| `Final_Autumn_2026-1.ipynb` | **The submission.** 107 cells (68 code), executed. |
| `METHODOLOGY_LOG.md` | The running justification of every analytical choice. It has sections A–J: §C Problem 1, §D Problem 2, §E Problem 3 (data traps, hypotheses, the 23 design decisions, the 38-specification ledger, results, referee review, final audit), §F rejected alternatives, §H dated deviations. |
| `PROJECT_PLAN.md` | The approved plan for all three problems, with lecture citations and an errata section at the end. |
| `LECTURE_METHODS_P3.md` | 152 rows: every method in L1–L8, whether Problem 3 uses it, and why or why not, citing printed numbers. §6 holds the proposals P-A … P-S and their status. |
| `AI_Coding_Guide.pdf`, `Lecture_1.pdf` … `Lecture_8.pdf` | Course materials. |
| `notes/` | Working materials (see `notes/README.md`): lecture text extractions (grep-able, PDF pages) and slide-cited notes, the data profile, the plans and competing P3 designs, the referee and final-audit reports, the eight per-lecture catalogues, and `notes/p3_build/` (the Problem 3 cell sources, a 1–2 minute development runner and the build scripts). |
| `*.csv` | Data. Problem 1: `Social_Network_Ads.csv`. Problem 2: `dj30.csv`. Problem 3: `asset_panel.csv`, `asset_info.csv`, `asset_returns_wide.csv`, `macro_global.csv`, `macro_country.csv`, `macro_country_extended.csv`. |

**Git.**
- Work on branch **`financial_analytics_opus`** (GitHub `Teknado/UChicago`) and push with
  `git push -u origin financial_analytics_opus`.
- Do not open a pull request unless the student asks.
- Commits before `c31b023` belong to an earlier, separate effort; ignore them.

**The notebook is the source of truth.** Problem 3's cell sources and build scripts are saved in `notes/p3_build/`,
and they matched the notebook exactly on 2026-09-28. Either edit the notebook's cells directly, or edit those sources
and rebuild with `notes/p3_build/build_nb.py` (which clears the Problem 3 outputs, so the notebook must then be
re-executed); never mix the two. `python3 notes/p3_build/dev_all.py`, run from the repository root, runs all Problem 3
cells in 1–2 minutes from a saved ledger, which is useful for testing a change before the full 11-minute run.
To find what a lecture says, `grep` `notes/lectures/Lecture_N.txt` (pages delimited by `===== PAGE n =====`).

## 5. Notebook map (0-based cell indices, current)

| section | cells |
|---|---|
| Exam instructions ("every number must come from a cell") | 1 |
| Problem 1 | 3–24. Exam question cells: 3, 5 (1.1), 10 (1.2), 14 (1.3), 18 (1.4), 22 (1.5). Setup: cell 4. |
| Problem 2 | 25–43. Exam question cells: 25, 27 (2.1), 29 (2.2), 33 (2.3), 39 (2.4). Setup: cell 26 (`SEED = 2694`). |
| Problem 3, exam text | 44 (intro), 45 (data and lag rules), 46 (setup code, the exam's own), 47 ("Two rules"), 48 ("A suggested workflow"), 49 ("What to submit") |
| §3.0 design, fixed before fitting | 50 (markdown), 51 (constants, the 38-specification ledger, library versions), 52 (power) |
| §3.1 Know your data | 53–64 |
| §3.2 Features | 65–68 |
| §3.3 Benchmarks first | 69–71 |
| §3.4 Harness (one forward-only splitter) and unit tests | 72–76 |
| §3.5 Ledger results | 77–88 |
| §3.6 Macro, placebo, robustness | 89–93 |
| §3.7 Portfolios | 94–100 |
| §3.8 Leakage audit | 101–102 |
| §3.9 Table 3.31, the headline numbers | 103–104 |
| §3.10 Deviations | 105 |
| Write-up, research-paper form (~6,100 words) | 106 |

**Data path.** Cells 4, 26 and 46 keep the exam's server path `/classes/41210_MiF_fall2026/Data/` and then add
`import os; _DATA_DIR = _DATA_DIR if os.path.isdir(_DATA_DIR) else './'`, so the notebook also runs with the CSVs next
to it.

## 6. What each problem did, and its answers

**Problem 1** (trees and ensembles):
- 5-fold stratified CV (the exam's `cv5`); the baseline "nobody purchases" scores 64.25%.
- CV prefers **3 leaves** (90.75%; ties with 4, and the tie goes to the smaller tree). The unpruned tree scores 85.00%.
- A forest (300 trees) scores 88.75% and boosting 89.00%: neither beats the small tree. The comparison is fold by fold,
  on shared folds.
- Importance (1.4): MDI ranks Salary first, held-out permutation ranks Age first.
- 1.5: why shuffled K-fold is wrong for time series.
- Shuffled folds are legitimate *here*: the rows are 400 unrelated people.

**Problem 2** (a VaR desk refitting on its own simulated scenarios):
- One desk ends at σ̂² = 1.405 × 10⁻⁵.
- Across 1,000 desks the median σ̂² is 0.0059, and the mean reported VaR is 0.775σ against a true 2.326σ. 57.8% of
  desks report less than a tenth of the truth.
- The drift of log σ̂² is about −1/(n−1) per night.
- Keeping the real days stabilises the estimate (band 0.94–1.06). Real returns have excess kurtosis 24.1; the
  synthetic scenarios lose the tails.

**Problem 3** (the main project):
- **Design.**
  - Target $y = r/\hat\sigma_{t-1}$, with σ̂ the 36-month volatility.
  - Features: within-class monthly ranks of the lagged characteristics `x1`–`x5`; global macro `x6`–`x9` as trailing
    z-scores × class; country macro (`x86`, `x12`–`x16`) as differentials vs country 7, trailing z-scores per country,
    × class for B, C, D. The macro lag is 2 months.
  - Training block 2003–2010; out-of-sample 2011–2024 (168 months).
  - Schemes: static, expanding (primary; 14 December refits) and rolling (96 months), each tuned on 3 forward annual
    validation folds.
  - Models: OLS, ridge (primary), lasso, PCR, a random forest with leaf 200 (our choice; L8 p.44 grows trees deep), a
    deep forest with leaf 5, and per-class ridge: 38 pre-listed specifications, Bonferroni bar 3.27.
  - Inference: a month bootstrap (B = 10,000; forecasts held fixed; our construction).
  - Placebo: macro circularly shifted 36–120 months.
  - Portfolios: P1 ∝ ŷ/σ̂ and P3 (within-class long–short), against EW, RP ∝ 1/σ̂ and TSMOM ∝ sign(R12)/σ̂, net of
    10 bp, with α regressions.
- **Data traps found.**
  - `x10` is a same-year average (look-ahead), so `x86` is used instead.
  - `x11` stops early for three countries, so it is excluded (gaps, and near-redundancy with `x86 − x12`).
  - Back-fills end in 2002-07.
  - The extended file has duplicates, hidden global series and low-frequency columns of unknown timing, so only `x86`
    is used.
- **Pre-registered answers: all NO.**
  - **Q1:** ridge M2 R²_OOS = 0.042% (SE 0.241%) vs the pooled mean; −0.279% vs zero. The characteristics add −0.045%
    (SE 0.027%) to class means.
  - **Q2:** macro changes R²_OOS by −0.123% (SE 0.085%) and ranks 4th of 9 against the placebos.
  - **Q3:** P1's net Sharpe is 0.44, against EW 0.28, RP 0.30 and TSMOM 0.12, but α = 0.61% a year with t = 1.74. P1 is
    almost the class-means portfolio (weight correlation 0.9994), and its gain comes from re-estimating class means
    each year: frozen at 2010, it earns 0.27.
- **Post-hoc additions, approved after the audit** (proposals P-A, P-B, P-C, P-D, P-E, P-G, P-H, P-L, P-M). They are
  labelled cells in the notebook, logged in §3.10 item 8 and in log §E.5:
  - P-A: the macro variables are described.
  - P-C: the SEs are about 19% too small because of serial dependence (the block bootstrap is our construction).
  - P-D: even a hindsight-chosen penalty does not beat class means.
  - P-G: PCR's components are macro factors.
  - P-H: the forest does not beat ridge, so there is no exploitable nonlinearity.
  - P-E: characteristic × macro interactions add nothing (a 39th test; Bonferroni 3.28).
  - P-B and P-L: the α's significance hinges on 6 outlier months (reported, not used).
  - P-M: the target's variance by class and year.
- **Proposals not run** (declined or low value): P-F boosting, P-I information criteria, P-J clustering, P-K
  AUC/direction, P-N out-of-sample terciles, P-O–P-S.

## 7. Conventions and gotchas

- **Citations** use **PDF page numbers**, as in `L5 p.57`. In Lectures 1, 5 and 8 the number printed on a slide can
  be lower, because of overlays (in L8, from PDF p.14 on, the printed number is 4 lower).
- **Scope details worth remembering:**
  - KNN regression is not taught (L6 teaches the KNN classifier).
  - PCR = PCA (L1) + OLS, combined by us.
  - The month bootstrap without refitting is our construction; L2 p.80 refits.
  - The Diebold–Mariano test and HAC SEs appear in no lecture; "robust" SEs are only named on L2 p.80.
- **Environment.** This machine: Python 3.11, pandas 3.0.6, scikit-learn 1.9.1. The course server: Python 3.14,
  pandas 3.0.5, scikit-learn 1.9.0. Known traps:
  - `LogisticRegression(penalty=None)` is deprecated; use `C=np.inf`.
  - `GradientBoosting*(n_iter_no_change=...)` uses a **shuffled** internal split: never use it on time series.
  - The `scoring=None` defaults use R² against the test mean; always set `scoring` explicitly.
  - `RandomForestRegressor` defaults to `max_features=1.0`, which is bagging.
  - pandas 3 uses `StringDtype`, `datetime64[us]` and copy-on-write.
  - The exam's setup cell silences warnings.
- **Style of the notebook.**
  - One step per cell. Asserts guard every leakage-sensitive step.
  - Figures use the palette blue `#2a78d6`, orange `#eb6834`, aqua `#1baf7a` and yellow `#eda100`, on single axes;
    `sns.set()` is never used.
  - Tables are numbered 3.x and figures Figure 3.x; the write-up cites them by number.
- **Pre-registration.** The §3.0 design cell is the pre-registered design. Do not change its rules. Any later change
  goes in §3.10 and log §H, and new analyses are labelled post hoc.

## 8. How to change the notebook safely

1. Edit the notebook JSON directly: cell `source` fields, with `json` or `nbformat`. Keep `indent=1` and
   `ensure_ascii=False` when saving, to preserve the format.
2. After any code change, **re-execute the whole notebook from the top** (about 11 minutes, 4 cores). Either run
   `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 Final_Autumn_2026-1.ipynb`,
   or use `nbclient.NotebookClient(nb, timeout=3600, kernel_name='python3').execute()` with the repo as the working
   directory. Then check that no cell raised an error and that every code cell has an execution count.
3. **Re-check the numbers.**
   - Every number in the write-up (cell 106) and in every answer cell must appear in a printed output, up to rounding.
   - Compare Table 3.31 before and after a change: unexpected changes to earlier values mean something broke.
4. Update §3.10 and `METHODOLOGY_LOG.md` §H for any change of method or result. If a method's status changes, update
   `LECTURE_METHODS_P3.md`.
5. Commit with a clear message and push to `financial_analytics_opus`. Commit messages end with the attribution lines
   the harness supplies.

## 9. Possible next steps (only if the student asks)

- Trim the write-up (about 6,100 words including tables) if the student wants it shorter.
- Run any of the declined proposals. They are post hoc, and each adds a test.
- Final proofreading of the answer cells, whose sentence limits are set by the exam: 1.2 one or two sentences;
  1.3 two or three; 2.1 two or three; 2.3 and 2.4 one paragraph each.
