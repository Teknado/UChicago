# Shared context for the final audit (read this first)

## The project
A student's final exam for BUSN 41210 Financial Analytics (Chicago Booth, Autumn 2026). AI assistance is explicitly permitted.
The exam notebook has 3 problems; Problem 3 (60 of 100 points) is an open-ended research project on cross-country asset
return prediction. The student's standing rules:
1. The **AI Coding Guide** is binding. Fatal errors: a shuffled split / fold on time-ordered data; a scaler, imputer or PCA
   (any transformation) fitted before the split. Also: `LogisticRegression()` is L2-penalised by default; `r2_score` uses the
   test-set mean, so it is not the course's R²_OOS.
2. **Scope = the 8 lectures.** Every model, formula and approach must be grounded in the lectures. A method that a lecture
   only *names* in passing (e.g. neural nets, SVM, elastic net, OOB error, SHAP, XGBoost, HAC SEs, t-SNE) was treated as
   NOT taught. Anything not taught must be labelled as "our construction/choice" or "[EXAM-DEFINED]".
3. Every number in an answer or in the write-up must be printed by a notebook cell (exam cell 1).

## Files (repo = /home/user/UChicago; do NOT modify anything in the repo)
- `AI_Coding_Guide.pdf` (text: scratchpad/text/AI_Coding_Guide.txt)
- `Lecture_1.pdf` … `Lecture_8.pdf` (text: scratchpad/text/Lecture_N.txt, pages delimited by `===== PAGE n =====`;
  PDF page = slide page; when a formula, figure or code block matters, read the PDF page itself with the Read tool's `pages`)
- Earlier notes (secondary, may be incomplete): scratchpad/notes/Lecture_N.md
- The executed notebook: `Final_Autumn_2026-1.ipynb`. A plain-text export with every cell's source and text/table outputs:
  **scratchpad/audit/nb_text.md** (search for `######## CELL <i>`). Figures are not in the export (marked `[figure]`).
- `METHODOLOGY_LOG.md` (every decision and its justification; §E = Problem 3), `PROJECT_PLAN.md`.
- Data files in the repo (csv) may be read with pandas for spot checks; never write to the repo.

scratchpad = <scratch>

## Notebook map (0-based cell indices)
- Cell 1: exam instructions ("every number must come from a cell", etc.)
- Problem 1 (trees and ensembles): cells 3–24
- Problem 2 (training on your own output): cells 25–43
- Problem 3: exam text cells 44 (project intro), 45 (data description and lag rules), 46 (setup code, exam's own),
  47 ("Two rules that apply throughout"), 48 ("A suggested workflow"), 49 ("What to submit").
  Our work: 50 (§3.0 design, fixed before fitting), 51–52 (constants, specification ledger, power), 53–62 (§3.1 know your data),
  63–66 (§3.2 features), 67–69 (§3.3 benchmarks), 70–72 (§3.4 models/harness), 73–80 (§3.5 ledger results),
  81–84 (§3.6 macro, placebo, robustness), 85–90 (§3.7 portfolios), 91–92 (§3.8 leakage audit), 93–94 (§3.9 headline
  numbers, Table 3.31), 95 (§3.10 deviations), 96 (the research-paper write-up).

## Problem 3 in one paragraph
50 assets (classes A 26 commodity-like, B 8 currencies, C 9 equity indices, D 7 bonds; anonymised), 12 countries, monthly
2000–2024; 5 hidden characteristics x1..x5 (x2 = 12-month past return, x5 = 36-month vol, x1 ~ carry, x3 ~ value/reversal,
x4 ~ change in carry), global macro x6–x9, country macro x10–x16 plus an extended file. Target y = r / σ̂(36m, t−1).
Features: within-class monthly ranks of lagged characteristics; global macro trailing-z × class; country macro differential
vs country 7, trailing-z per country, × class. Training 2003–2010, OOS 2011–2024 (168 months), static / expanding (primary) /
rolling (96m) schemes, 14 December refits, 3 annual validation folds inside each window. Models: OLS, ridge (primary), lasso,
PCR, random forest (leaf 200) and deep forest (leaf 5), per-class ridge; 38 pre-listed specifications. R²_OOS vs the pooled
trailing mean (primary), zero, per-class and per-asset means; month-block bootstrap SEs (B = 10,000). Placebo = macro
circularly shifted 36..120 months. Portfolios: P1 ∝ ŷ/σ̂, P3 within-class long-short, vs EW, RP (1/σ̂), TSMOM
(sign(R12)/σ̂); 10 bp costs; α regressions. Verdicts (pre-registered): Q1 no, Q2 no, Q3 no. An independent referee already
reviewed the first run; its fixes are in (see METHODOLOGY_LOG §E.6 and notebook §3.10).

## Output conventions
- Cite evidence precisely: `cell 76, Table 3.12b`, `L5 p.57`, `AI guide §4b`.
- Severity: FATAL (a guide fatal error or a result-changing bug) / MAJOR (wrong result, false or unsupported statement,
  missing required element) / MINOR (imprecision, weak citation, presentation) / NIT.
- Separate clearly: (a) ERRORS in what exists, (b) OMISSIONS worth proposing (new analyses/methods), (c) things verified OK.
- Be concrete and sceptical. Do not pad. Do not invent page content: quote the slide when you rely on it.
