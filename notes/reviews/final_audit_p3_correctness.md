# Problem 3: final correctness audit (after the referee fixes, commit 848a344)

Scope: cells 50–96 of `Final_Autumn_2026-1.ipynb`, the source diff ee43a2e..848a344, and the referee report
(scratchpad/p3/referee_report.md). Nothing in the repo or in scratchpad/p3 was modified.

How I checked:
- I reran every Problem 3 cell from the scratchpad sources with the saved ledger (`audit/tmp/p3c_run_all.py`, log `p3c_run_all.log`).
  All 56 values of Table 3.31 match the executed notebook; only the runtime and the display format of 3.27 differ.
- I diffed every Problem 3 output between ee43a2e and 848a344. Every output that existed before is unchanged, apart from the
  runtimes and the added rows and tables. So "computations unchanged" (§3.10, item 5) is true.
- The executed notebook has 58 code cells, execution counts run 1..58, and no cell raised an error.
- Extra checks are in `audit/tmp/p3c_checks.py` and `p3c_checks2.py` (logs `*.log`).
- No fatal pattern was found: no shuffled or CV split, no transformation fitted before the split, no `r2_score`, no `LogisticRegression`.

---------------------------------------------------------------------------------------------------------------------------

## 1. Referee findings: status

| # | finding | status | evidence |
|---|---|---|---|
| M1 | "static tilt" attribution | **FIXED** (text has a residual overclaim, m2) | P1 on `ridge|M2|static` is now in Tables 3.24–3.26, 3.27 (α −0.16%, t −0.91) and 3.28b. The text now says re-estimated class means, and frozen 0.27 < RP 0.30. I checked the static P1: it equals P1 on the frozen 2003–10 class means (weight correlation 0.9999999, net Sharpe 0.2666 for both). |
| M2 | "halved in the second half" | **FIXED** (period mismatch left, m5) | Table 3.27b: −0.162% (t −0.37) and +1.358% (t 2.47). The text states these, the fall from 0.58 to 0.32, and that EW's 0.53 beats P1 in 2018–24. The hedge story is now hedged. |
| M3 | power / overclaim | **PARTLY** | The detectable effect of the headline statistic is fixed (0.48% / 0.68%; I verified q1_se = 0.24076%). But the paired test that replaced it is now over-sold as precise (MAJOR-2 below). |
| M4 | three false statements | **FIXED** | Table 3.12b added; I verified its signs. The "−0.4..+0.2" range is replaced by "29 of 38 in −0.97..+0.21, nine others listed", and I verified the list. "0.12–0.30" is corrected. "Rolling vs expanding < 1 SE in every model" is true: \|t\| ≤ 0.88. |
| M5 | per-asset benchmark missing | **FIXED** (one sentence inverted, MAJOR-3) | Table 3.10c: M1 0.360% (t 2.18), ridge M2 0.316% (t 1.92); 17 of 38 positive. |
| M6 | "why so little" logic | **FIXED** (residual m4) | Table 3.18b (20–80%). The x2/x5 cross-sectional argument is rewritten. |
| m1 | placebo wrap | **FIXED** | Table 3.19b plus rebuild check. I verified all 8 shifts, not only 36 and 120: every target month ≤ s+60 changes by ≥ 4.3e−2, and none after it changes by more than 1.8e−11. |
| m2 | PCR K = 0 | **FIXED** | −0.095% (8 of 14 refits pick 0). K = 0 gives zero slopes in `LinearFE`, and the 8 K = 0 refits reproduce the M1 forecasts exactly. |
| m3–m8 | data claims, citations | **FIXED** | x1/x4 limited to class B; x11 wording; "only 0.042→0.026"; extended-file timing; AMP 2013 cited for value and momentum only; L5 p.57 quote verified. |
| m9 | weak citations | **FIXED** | leaf 200 is flagged as a departure; PCR and the month bootstrap are labelled "our construction"; L7 p.23 and L5 p.23 are removed. L8 p.40 (the variance of a deep *tree*) now backs "deep forests overfit". That is acceptable but weak (NIT). |
| m10 | stale cell numbers | **FIXED** | The only reference left is "cell 1", which is correct. |
| m11 | risk-budget description | **FIXED** | Table 3.28: identical class shares, as the text now says. |
| m12 | "one-way" turnover | **FIXED** | The residual `turnover()` docstring in cell 69 still says "One-way traded notional" (NIT). |
| m13 | audit counts | **FIXED** | 326 / 100 / 86, which I verified by hand. |
| m14 | vacuous unit test | **FIXED** | |
| m15–m18 | sign stability, lag asymmetry, rolling P1, pre-registration evidence | **FIXED** | |
| m19 | untried extensions | **FIXED** | The exam's "widened panel" is not named (NIT). |
| m20 | tercile levels | **PARTLY** | Table 3.6b gives bottom, middle and top mean y, but pooled over classes. There is no per-class or return-unit version, and the write-up never discusses it. |
| m21–m22 | Methodology reproducibility; write-up inserted | **FIXED** | |
| m23–m26 | server robustness | m24 disclosed; the rest need no action | |

## 2. Code added after the review: audit

All correct, and none leaks.
- **3.6b** uses the training block only (the assert is kept) and the same tercile construction as 3.6.
- **3.10c** matches the per-asset column of 3.10b.
- **3.12b** calls `delta_boot(static or rolling, expanding)`, which gives R²(scheme) − R²(expanding); I verified the sign against Table 3.10.
- **3.18b** is descriptive and uses predictors only.
- **3.19b** and `touched_months`: the analytical count t ≤ s + 58 + MACRO_LAG is right (the window runs over raw months m−61..m−2).
- **The tolerance change** (1e−12 → 1e−8) has about 9 orders of magnitude of margin on both sides (round-off ≤ 1.8e−11; genuine changes ≥ 4.35e−2). The assert checks only the max for two shifts; I checked every month for all 8.
- **The `grid=` argument** is backward compatible, and `fixed=` takes precedence. Ties go to K = 0 because it is first in the grid. The ledger outputs are unchanged.
- **Static and rolling P1** use σ̂ at t−1, and no forecast uses data after its fit window.
- **3.27b and 3.28b** reproduce, and `tilt_tab.iloc[:, 2]` is the primary P1 column. Positional indexing is fragile (NIT).
- **The new rows of Table 3.31** all reproduce.

## 3. Write-up (cell 96) and §3.0–§3.10 markdown: flagged statements

### MAJOR
1. **"Macro lowers R²_OOS in every model"** (Preview; §4.2 "Every model and scheme loses when macro is added"; §5 "macro loses in every model"; Conclusion "makes every model slightly worse").
   - A table contradicts this. `ridge|M3-x10|expanding` has all the macro inputs, with x10 in place of x86, and scores 0.065% against ridge M2's 0.042% (Table 3.10b). Paired, the difference is +0.023% (SE 0.036%, t 0.63; my computation).
   - "Slightly" is also wrong for OLS (−19.5%, −4.7%, −5.3%), rf static (−3.0%) and rf-deep (−3.9%) (Table 3.18).
   - *Fix:* add the M3-x10 and M3-lag1 pairs to Table 3.18. Say "in all 14 pairs of Table 3.18 … the only macro variant above ridge M2 is curated x10 (+0.02%, SE 0.04%)". Drop "slightly".
2. **The paired "sharpest test" is presented as precise** (§1 "sharpest within-class test"; §4.1 "precisely estimated, and it is below zero"; §5 "even the sharpest paired test can only rule out effects of a few hundredths of a percent"; Conclusion "precisely estimated").
   - The SE of 0.027% is small because ridge shrank the characteristic part to almost nothing. It sits at the no-signal end in 64% of refits. The characteristic part of the forecast has an SD of 0.009, against 0.038 for OLS.
   - On the same months, OLS M2 − M1 = −0.102% with SE 0.075%, 2.8 times larger. The §3.0 power calculation implies an SE of about 0.09% for a true 0.3% within-class signal.
   - So the paired test shows this model's characteristic part was not helpful (t −1.67). It does not rule out a within-class R² of about 0.15–0.2%.
   - *Fix:* print the OLS M2 − M1 paired delta, and rephrase as "a test of the sign of the fitted characteristic component; its SE reflects the shrinkage".
3. **"The per-asset trailing mean is a harder bar in the other direction, because it is noisy"** (Methodology). This is inverted. It is the *easiest* bar: it loses 0.27% to the pooled mean (Table 3.8), and M1 beats it by 0.36% (Table 3.10c). *Fix:* "an easier bar".

### MINOR
1. **"the one result beyond 2 SE, against the per-asset mean"** (§5). It is not the only one:
   - x10 vs x86: +0.146% (SE 0.069%), t 2.09 (Table 3.21);
   - the 2018–24 α: t 2.47 (Table 3.27b);
   - against the per-asset mean, ridge M2-zscore (t 2.14) and M3-x10 (t 2.07) also exceed 2 SE (my computation; 3.10c prints only 3 of the 38).
   - *Fix:* list them, or say "the one *forecast* result".
2. **Causal attribution on differences that are not significant**: "The gain comes from class means re-estimated every year"; "Where the gain comes from"; "Frozen…, the tilt does worse than risk parity"; "…beat risk parity; the same tilt frozen in 2010 did not".
   - Paired bootstrap (my computation): P1 − P1 static = 0.178 (SE 0.109, t 1.64); P1 static − RP = −0.033 (SE 0.036, t −0.92); P1 − RP = 0.14 (SE 0.11).
   - *Fix:* print these, and say "is associated with" / "point estimates".
3. **"class intercepts help currencies (+1.8% …, because currencies earn about zero)"**.
   - +1.8% is ridge M2's figure; M1 gives 1.92% (Table 3.11).
   - The OOS mean y of class B is −0.109, which is clearly negative, and no cell prints it.
   - *Fix:* "+1.9%", and print the class OOS means.
4. **"P3 is exactly that bet"** (the x2/x5 rank bet).
   - P3 is the within-class deviation of the shrunk ridge forecast, a mix of all five ranks. The mean coefficients are u5 −0.0034, u3 0.0008, u1 0.0007, u4 0.0006, u2 −0.0001 (Table 3.15).
   - *Fix:* "P3 is the within-class bet on the characteristic forecast (mostly u5)".
5. **"It cannot explain its α, which rose in the same years"** compares different periods: the correlations are for 2011–19 / 2020–24 (Table 3.30), the α for 2011–17 / 2018–24. With matched periods, the C–D correlation moves from −0.23 to +0.14 (my computation), so the argument survives. *Fix:* print the matched-period correlation.
6. **"Monthly returns are nearly serially uncorrelated"** (Limitations) is not printed. The AC(1) across assets has median 0.012 and ranges from −0.13 to +0.32 (my computation). *Fix:* print it, or qualify the claim.
7. **"In raw-return units the only sizeable number is the equity-like class"**. Table 3.13 also shows class C vs the asset mean at 1.4–1.6%, and rf M3 class B at −3.25% / −4.61%. *Fix:* "the largest positive number".
8. **"permutation importance ranks the class dummies and volatility first"** holds for rf M2 only. For rf M3, g_x7 ranks first (0.167%, Table 3.16).
9. **Rounding:** the best placebo gain is quoted as "+0.13%". The unrounded value is 0.1247%, which rounds to 0.12%; "+0.13%" comes from rounding Table 3.19's printed 0.125% a second time.
10. **Tercile sentence:** the third-strongest spread is x2 in class D (t −2.10), not "x4 across classes (t 2.08)" (Table 3.6).
11. **Discussion item 3, "That is why every model gets worse, and why the placebos do as well"**, is causal and repeats the "every" error. *Fix:* "consistent with estimation error (in-sample R² rises, out-of-sample R² falls, Table 3.10b)".

### NIT
- "fat tails" for class A is not printed; class A's median excess kurtosis (1.34) is no higher than class C's (1.38).
- The "2021–23 inflation shock" is not established by any cell.
- "0.6 SE", "about 2 SE" and "20-fold" are derived, not printed; the header says every number is printed.
- "A model with class intercepts only produces the same portfolio": say "almost the same (0.9994)".
- "a post-hoc split into two sub-periods": the sub-periods were fixed in §3.0; only the regression is post hoc.
- Cell 53, "Every data fact quoted in the write-up is printed here": Tables 3.8 and 3.30 are printed later.
- Cell 81, "for s ≥ 72, the first out-of-sample years": for s = 72 only one OOS month is affected (Table 3.19b).
- Cell 91, "Nothing is imputed": the levels are filled with 0 for their first 24 months, and so are the 2003-01 rows of the x10 variant (§3.10, item 2).
- The cell 69 docstring still says "one-way".

## 4. Verified correct

These all match their printed sources:
- **Numbers.** Every other number in cell 96 matches Tables 3.1–3.31 at the printed precision. That includes 0.94 (0.9354), 26% (0.2650), 34% (0.3450), 0.39 (0.3946), −0.16% twice (−0.1595 and −0.1617), 0.68% (0.684), 0.61%, t 1.74, 0.9994, 45/74/46%, 0.169/0.216/0.125, 0.122/−0.011, 260/363 bp, 29 + 9 specifications, 17 of 38, 8 of 14, and 1 / 49.
- **Other claims in cell 96:**
  - the L5 p.57 and L8 p.62 quotes;
  - "every positive value inside one SE";
  - "static worst for every model";
  - "no spec beats zero";
  - "Lasso: no variable in half the refits";
  - "PCR at K = 1 in 12 of 14".
- **Cells 50 and 95** are consistent with the outputs: the decision rules are applied as written, and the "Nothing else changed" list is true.
