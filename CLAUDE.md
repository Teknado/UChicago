# Context for Claude: BUSN 41210 Financial Analytics final exam (Chicago Booth, Autumn 2026) — submission branch

This branch, `financial_analytics_final`, holds the **submission** and nothing else of the working process.

| file | what it is |
|---|---|
| `Final_Autumn_2026-1.ipynb` | **The submission.** The exam notebook filled in place: the template's cells 0–38 in their exact order, then Problem 3 split into section cells (70 cells: 22 code, 48 markdown), executed top to bottom with no errors (about 10–16 minutes). |
| `METHODOLOGY.md` | The full explanation of what was done and why: every rule, every choice, the alternatives and why each was used or set aside, the answers. Every result it quotes is printed by the notebook. |
| `AI_Coding_Guide.pdf`, `Lecture_1.pdf` … `Lecture_8.pdf` | Course materials. |
| `*.csv` | Data (Problem 1: `Social_Network_Ads.csv`; Problem 2: `dj30.csv`; Problem 3: `asset_panel.csv`, `asset_info.csv`, `asset_returns_wide.csv`, `macro_global.csv`, `macro_country.csv`, `macro_country_extended.csv`). |

**The development history lives elsewhere and must not be brought into this branch**: the 107-cell analysis notebook, the
methodology log, the plans, the lecture catalogues and the build and verification scripts are on
`financial_analytics_opus`; the comparison with another AI's attempt is on `financial_analytics_review`. The submission
notebook is built from `Final_Autumn_2026_final.ipynb` on `financial_analytics_opus` by `notes/final_clean/build_clean.py`
and checked by `notes/final_clean/verify_clean.py` (both on that branch).

## The student's standing rules (always follow them)
1. **The AI Coding Guide is binding.** Fatal: a shuffled split or fold on time-ordered data; a scaler, imputer, PCA or
   other transformation fitted before the split. `LogisticRegression()` is L2-penalised by default; `r2_score` is not the
   course's R²_OOS.
2. **Scope = the 8 lectures.** A method a lecture only names counts as not taught. Anything not taught is labelled "our
   construction" or "our choice"; exam-defined quantities follow the exam exactly.
3. **Every number must come from a cell.** Problem 3's Table 3.31 prints every number the write-up quotes.
4. **Plan first, then ask; do not code until the student approves.** Fix outright errors directly; propose any new model
   or analysis and wait for approval.
5. **Honest reporting.** Negative results are results; never switch a benchmark after seeing results; put the search size
   next to every headline; label anything added after results were seen as post hoc.
6. **This branch contains no reference to the working process** (reviews, audits, versions, other attempts, proposal
   codes). Keep it that way in any edit: the notebook and `METHODOLOGY.md` describe the work, not its history.
7. The Problem 2.1 answer is the student's own prediction, written before any Problem 2 code. Never edit it.
8. The exam's sentence limits: 1.2 one or two sentences; 1.3 two or three; 2.1 two or three; 2.3 and 2.4 one paragraph each.

## Key decisions (do not reopen without asking)
- `SEED = 2694` for Problem 2; Problems 1 and 3 use 7034.
- Problem 3: pooled models with class-specific terms, plus PCR and per-class models; primary benchmark the pooled trailing
  mean; costs 10 bp per unit of turnover Σ|Δw| with a 0–50 bp grid and break-evens; expanding window primary.
- The answers: Q1 no detectable predictability; Q2 macro adds nothing; Q3 no forecast-driven outperformance (the gain is a
  re-estimated class-premium tilt). See `METHODOLOGY.md` §4.

## Notebook map (0-based cells)
Exam text 0–5, 8, 11, 14, 17, 20, 22, 24, 27, 30, 33–38. Problem 1: code 6, 9, 12, 15, 18; answers 7, 10, 13, 16, 19.
Problem 2: setup 21 (`SEED`); answer 2.1 in 23; code 25, 28, 31; answers 26, 29, 32. Problem 3: setup 35 (the exam's);
introduction 39; Sections 3.0–3.9 as (note, code) pairs in 40–59 (code cells 41, 43, …, 59; each prints a banner);
Section 3.10 (deviations) in 60; **write-up in 61–69**, one markdown cell per section of the paper. Each problem runs on its
own in a fresh kernel (Problem 1 needs the exam's import cell 2).

## Changing the notebook safely
1. Edit cell `source` fields with `json` (keep `indent=1`, `ensure_ascii=False`).
2. After any code change, re-execute the whole notebook from the top with the repository root as working directory
   (`nbclient`, timeout 3600), and check that no cell errors and every code cell has an execution count.
3. Every number in an answer or in the write-up must appear in a printed output; compare Table 3.31 before and after.
4. Any change of method or result goes in Section 3.10 (Deviations) and in `METHODOLOGY.md`, labelled post hoc if it
   follows the results.
5. Environment: Python 3.11, pandas 3.0.6, scikit-learn 1.9.1 here; the class server has Python 3.14, pandas 3.0.5,
   scikit-learn 1.9.0. Traps: `GradientBoosting*(n_iter_no_change=...)` shuffles internally; `scoring=None` defaults use
   the test mean; `RandomForestRegressor` defaults to bagging; pandas 3 uses copy-on-write and `datetime64[us]`.
6. Commit with a clear message and push to `financial_analytics_final` (`git push -u origin financial_analytics_final`).
   Do not open a pull request unless the student asks.
