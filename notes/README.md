# notes/: working materials behind the submission

None of this is part of the submission, which is `Final_Autumn_2026-1.ipynb`. These are the materials used to build and
check it, kept so that a later session does not have to redo them. The decisions themselves are recorded in
`METHODOLOGY_LOG.md`, `PROJECT_PLAN.md` and `LECTURE_METHODS_P3.md` in the repository root.

## Contents

| folder / file | what it is | caveats |
|---|---|---|
| `lectures/Lecture_N.txt`, `lectures/AI_Coding_Guide.txt` | Text extracted from the PDFs, one block per page, delimited by `===== PAGE n =====` (PDF page numbers, as all citations use). Fast to search with `grep`. | Formulas, figures and code are sometimes garbled; read the PDF page when they matter. |
| `lectures/Lecture_N_notes.md` | Slide-cited notes on each lecture, written at the start of the project. | Secondary: the catalogues below are more complete. |
| `lectures/lecture_digest.txt`, `lectures/exam_relevance.txt` | A topic index of each lecture, and each lecture's relevance to each exam question. | Written at the planning stage. |
| `data_profile.md` | The pre-modelling data profile: shapes, traps (`x10` look-ahead, `x11` stops, back-fills, duplicates), characteristic identifications, macro descriptions. It never relates a predictor to a future return. | `<scratch>` refers to the old temporary workspace. |
| `plans/` | The per-problem plans; the three competing Problem 3 designs (A: linear first, B: nonlinear, C: evaluation first) and the judge's scoring; the merged P3 plan. | The approved plan is `PROJECT_PLAN.md` in the root. |
| `reviews/p3_referee_report_first_run.md` | The independent referee's report on the first full Problem 3 run. | All findings addressed; see log §E.6. |
| `reviews/final_audit_*.md` | The final audit against the AI Coding Guide, of Problems 1–2, of Problem 3's coverage of the exam, and of Problem 3's correctness. Also the corrections applied and the proposals P-A … P-S. `reviewer_brief.md` is the brief the reviewers were given. | Findings addressed; see log §E.7. Cell indices refer to the notebook *at that time* (it has since grown by 10 cells), so use section and table numbers instead. |
| `lecture_catalogs/L1.md` … `L8.md` | One reviewer per lecture: every method in the lecture, its status in Problem 3, citation checks and candidate additions. | They contain "auditor spot check" numbers that the notebook does **not** print, so they cannot be quoted in the submission. `LECTURE_METHODS_P3.md` in the root is the cleaned, consolidated version. |
| `p3_build/` | The Problem 3 cell sources and the tooling used to build and check them (see below). | |

## `p3_build/`: how Problem 3 was built

- `cells_a.py` … `cells_h.py` hold the Problem 3 cells in notebook order, as Python strings: §3.0–3.1, §3.2, §3.3,
  §3.4, §3.5, §3.6, §3.7, and §3.8–3.10. `cells_x.py` holds the nine post-hoc checks, which are spliced into those
  lists. `writeup.md` is the write-up (the notebook's last cell).
- **They match the committed notebook exactly** (checked cell by cell on 2026-09-28). The notebook is authoritative.
  If the notebook is edited directly, these sources go stale, so either keep editing the notebook, or edit the sources
  and rebuild, never a mix of the two.
- `nbtools.py`: `load()` and `save()` preserve the notebook's JSON format; `execute_prefix(nb, upto)` executes cells
  0..upto with nbclient in the repository directory and writes the outputs back.
- `runner.py` and `dev_prefix.py`: a fast development runner. It executes cell sources in one shared namespace and
  saves figures to `p3_build/fig/`.
- `dev_all.py`: runs every Problem 3 cell from the sources in about 1–2 minutes. It loads `ledger_results.pkl` instead
  of refitting the 38 specifications. **Run it from the repository root**: `python3 notes/p3_build/dev_all.py`.
- `ledger_results.pkl`: the saved outputs of the ledger cell for the 38 pre-registered specifications (`RES`, `LOGS`,
  `IMPORTANCE`, `LEAK_YH`, `LEAK_LOG`, `RUNTIME`). It was pickled with pandas 3.0.6 and numpy 2.4.6; if it will not
  load, rerun the notebook's ledger cell and re-pickle those objects.
- `build_nb.py`: **replaces** the notebook's Problem 3 cells, the setup cell and the write-up with these sources, and
  clears their outputs. Afterwards the whole notebook must be re-executed (about 11 minutes), for example:
  `python3 -c "import sys; sys.path.insert(0,'notes/p3_build'); import nbtools as T; nb=T.load(); last=max(i for i,c in enumerate(nb['cells']) if c['cell_type']=='code'); T.execute_prefix(nb, last, 3600); T.save(nb)"`
- `check_numbers.py`: lists every number in the write-up that does not appear verbatim in a printed output. Expect only
  roundings (for example −1.37% quoted as −1.4%) and section headings such as 4.2.
