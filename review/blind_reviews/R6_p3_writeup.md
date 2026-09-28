# R6: Problem 3 write-up, graded as a research paper (blind)

Scope: the final markdown cell of each notebook (A: cell 40, 3,082 words; B: cell 106, 6,084 words). Each is checked against the exam's "What to submit" (cell 38) and the rule in cell 1 that every number must come from a cell.

Method: all code-cell outputs of each notebook (stream, text/plain, text/html) were pulled out of the .ipynb into `reports/p3write_work/solution_{A,B}.ipynb.outputs.txt`. Every numeric token in each write-up was then matched against the output tokens, allowing rounding and the fraction-to-percent conversion (scripts `prov.py`, `strict.py`). The matching is permissive: a short number such as 0.85 can match an unrelated output such as a P1 accuracy. So every hit on a substantive number was then checked by hand in context. Years, section numbers and small integers were excluded.

---

## (a) Number provenance

### Solution A

**Numbers verified against printed output, in context (correct):**
- Universe: 50 assets, 26/8/9/7, 12 countries, 300 months, 15,000 asset-months (`Panel shape: (15000, 10)`); 96-month training block, 168 OOS months, 8,400 asset-months, 14 refits.
- Class means +6.02%, −0.24%, +4.77%, +2.68% (printed Table 1).
- Correlation block matrix 0.211 / 0.240 / 0.176 / −0.080 / 0.550 / 0.230 / 0.053 / 0.725 / −0.104 / 0.580 (printed Table 2).
- x2 AR(1) 0.916 (printed 0.9157).
- x10: 300/300 country-years, corr 0.998 (0.9981). x11: 99 missing, corr 0.9882. asset_16 floor 0.004.
- R²_OOS for M1, M2 ridge, M2 RF and M3: +0.087/−0.234, −0.001/−0.322, −0.115/−0.437, −2.972/−3.303 (printed Table 5).
- By-class values +0.054/+2.026, −0.166/−0.779, −0.306/−0.307, −0.346/−1.464 (printed Table 6).
- The whole portfolio table: 2.51/8.92/0.282/0.281/0.000/−24.15; 1.64/5.34/0.308/1.61/0.302/0.022/−11.46; 0.80/4.04/0.199/0.50/0.123/0.251/−13.88; 1.80/3.97/0.454/1.69/0.426/0.089/−9.70 (printed Table 7).
- Attribution: R² 0.9028, α +0.52% (t 1.56), β_RP 1.1846 (23.13), β_EW −0.3134 (−10.36), β_TS 0.0861 (3.46) (printed Table 8).

**Numbers that CONTRADICT a printed output of the same notebook:**

| # | Write-up says | Printed output says |
|---|---|---|
| A-C1 | Table 1 placebo rows: 36m **−2.715% / −3.045%**, 48m **−3.084% / −3.415%**, 60m **−2.890% / −3.220%** | Table 5: `Placebo 36m Shift −2.972% −3.303%`, `Placebo 48m Shift −2.972% −3.303%`, `Placebo 60m Shift −2.972% −3.303%` (identical to M3) |
| A-C2 | Abstract: placebos give "(−2.7% to −3.1%)"; §4.1: "placebos (−2.71% to −3.08%)" | All three are −2.972%, the same as real macro |
| A-C3 | Class A "Sharpe ratio 0.194" | Table 1: A Sharpe 0.412 |
| A-C4 | Class B "Sharpe −0.023" | Table 1: B −0.030 |
| A-C5 | Class C "Sharpe ratio 0.282" | Table 1: C 0.327 |
| A-C6 | Class D "highest standalone Sharpe ratio (0.405)" | Table 1: D 0.508 |
| A-C7 | Worst months: A −54.65%, B −16.29%, C −25.36%, D −12.08%; A best +91.52% | Table 1 worst month: −19.85%, −8.52%, −16.02%, −5.03%; A best +13.90% (these are class-average series). The write-up's figures are per-asset extremes that A never prints. |
| A-C8 | §2.4 T3: "Assets 1, 2, 5, 21, 26" | Printed audit line: "Assets 1, 2, 5, 21, 26, 3" (and that string is hard-coded, see below) |
| A-C9 | §2.4: x11 "≡ x86 − x12 … exact linear combination … exact collinearity" | Printed corr 0.9882, which is not an identity |

**Numbers NOT printed anywhere in Solution A's notebook:**

| # | Number (context) |
|---|---|
| A-N1 | "annualized volatility from 6.6% in sovereign debt to 31.0% in commodities" (Intro) |
| A-N2 | Class volatilities ".97%", ".19%", ".95%", ".63%" (§2.1) and ".0%", ".6%" (§2.2). The leading digits were lost in rendering, and none of these is in Table 1 (printed 14.61 / 7.95 / 14.61 / 5.28). |
| A-N3 | "commodities dominating 80% of the squared-error loss" |
| A-N4 | x5 "ρ = 1.0000 with trailing 36-month SD" |
| A-N5 | x2 "0.9934 correlation with trailing 12-month compounded returns" |
| A-N6 | x3 "−0.93 to −0.98 correlation with trailing 60-month returns" |
| A-N7 | x4 "0.85 to 0.98 correlation with 12-month changes" |
| A-N8 | x1 carry, x3 "book-to-market in equities": no printed diagnostic at all |
| A-N9 | "VIX, skewness +2.06" |
| A-N10 | "static country means account for up to 69.4% of raw differential variance" |
| A-N11 | x11 stop dates "Oct 2020, Aug 2021, Mar 2024" |
| A-N12 | 4,800 training asset-months (arithmetic only) |
| A-N13 | Attribution table SEs 0.0033, 0.0512, 0.0303, 0.0249; p-values 0.121, 0.000, 0.001; F = 508.2. These can be back-derived from coef/t and R², but they are not printed. |
| A-N14 | "VIX spikes co-move with equity drops at ρ = −0.295" |
| A-N15 | Characteristic drivers of the class-C result ("intermediate momentum, carry, long-horizon reversals"): coefficients are collected in `coef_paths` but never printed or plotted |

**Printed, but from hard-coded strings rather than computation.** The write-up relies on these, so their provenance is hollow:
- "(Sweden, Japan, Switzerland)"
- "Assets 1, 2, 5, 21, 26, 3 … All terminate by August 2002"
- "43 consecutive months during BoJ YCC"
- "34 annual-stamped + 26 quarterly-stamped series"
- "PCA on 148 extended variables"
- "300/300 country-years"

### Solution B

- 664 numeric tokens were extracted. After excluding years, table/section numbers and small integers, every remaining unique number had a matching printed token.
- About 90 substantive numbers were then checked by hand in context: Table A, Table B, Table C, Tables 3.1a, 3.3, 3.5, 3.11, 3.12b, 3.13, 3.24, 3.27–3.28b, 3.27c–e, the power table, and the headline Table 3.31 printed by cell 104. All of them matched.
- Rounding cases, all correct:
  - EW net Sharpe printed 0.276 → "0.28"
  - OLS static macro loss −19.467% → "−19.47%"
  - deep-forest macro −3.928% → "−3.93%"
  - corr(x10, prior-year mean) 0.838612 → "0.839"
  - power t 3.338882 / 1.008815 → "3.34 / 1.01"
  - class A worst month −0.547 → "54.7% loss"
- **Numbers not printed:** none found.
- **Numbers contradicting output:** none found.
- One wording imprecision, not a numeric error: "static fit is worst … t = −2.47 for OLS". That is OLS M2. OLS M3 is −6.96, so the statement still holds.

---

## (b) Findings

| id | sol | severity | finding | evidence |
|---|---|---|---|---|
| A1 | A | FATAL | The write-up's results table contains placebo R²_OOS values that exist nowhere in the notebook and contradict the printed ones. It then draws its Q2 conclusion from the fabricated spread. The printed placebos are identical to real macro to three decimals because the placebo code is broken: it shifts `g_*` columns that are not in `m3_features`, since the interaction columns were built before the shift. So the placebo test was never actually run. | Write-up Table 1 "−2.715% / −3.045%" etc. vs printed `−2.972% −3.303%` ×3; code lines building `inter_glob` before `placebo_panels` |
| A2 | A | ERROR | Methodology misdescribes the placebo: it says "raw macro series are shifted … before feature engineering". The code shifts only the engineered global z-scores (not country macro), after the interactions were built. | §3.2 item 5 vs cell 39 placebo block |
| A3 | A | ERROR | Headline Q1 claim contradicts A's own numbers. (1) "Characteristics deliver resilient predictability within equities **and bonds**", but bonds are −0.306% vs class mean. (2) "genuine predictive content", but M1 class means (+0.087%) beat M2 ridge (−0.001%), so the characteristics subtract value. (3) M2 is labelled "Optimal Parsimonious Model". | Abstract item 1, §1 last para, Table 1 vs printed Tables 5–6 |
| A4 | A | ERROR | Class Sharpe ratios and worst months in §2.1 contradict the notebook's printed Table 1 (0.194 vs 0.412, −0.023 vs −0.030, 0.282 vs 0.327, 0.405 vs 0.508; worst months not printed). | see A-C3…A-C7 |
| A5 | A | ERROR | Over 25 further numbers are not printed anywhere, in breach of cell 1's rule. These include every characteristic-identification correlation and the attribution SEs, p-values and F. | A-N1…A-N15 |
| A6 | A | ERROR | Severe rendering corruption from escape and `$`-expansion damage. The notebook source holds control characters: 30× TAB (`	ext`), 8× backspace (`egin{pmatrix}`, `eta_`), 4× BEL (`lpha`), 2× form-feed (`rac{`), 7× CR (`\rho` → `ho`). R²_OOS appears as `^2_{OOS}$` 12 times (missing `$R`). Variable names are deleted: "driven primarily by intermediate momentum (\$), carry fundamentals (\$)"; §2.3 list items read "**\$ (36-Month Realized Volatility):**", so the reader cannot tell which x is which. Leading digits are lost: "(.97\%\$)", ".92\%", ".1\%", ".9\%", ".123\$", "floor of .004\$", " = 1.56\$", " = 508.2\$". Display equations appear as raw text, and 26 of 181 lines have unbalanced `$`, which garbles the tables. | raw cell 40 source in solution_A.ipynb |
| A7 | A | ERROR | Real-world identities are asserted as fact for anonymised data, with no evidence in the notebook: "commodities", "G10 currencies against USD", "Nikkei 225 … Lehman", "UK Gilt … mini-budget", "Japanese 10Y JGB", "BoJ YCC", "country_7 (US)", "Sweden, Japan, Switzerland", "VIX", "CFNAI". Event narratives (Brexit, Swiss unpegging) are offered as explanations but never tested. | §2.1, §2.4, §6.1 |
| A8 | A | ERROR | Overclaims on the portfolio evidence. "Decisively outperforming", "positive trend-following alpha", "Positive net timing alpha" are all said of α = 0.52% with t = 1.56 (p = 0.12). No SE is given for the Sharpe difference. β_RP = 1.18 with R² = 0.90 shows P1 is largely scaled risk parity, which the text half-admits. "Proves that macro conditioning acts purely as sampling noise" rests on a placebo that did not run. "By construction orthogonal to real-time market states" is also false for persistent series, as the exam itself warns. | Abstract item 3, §4.1 item 2, §5.3 table |
| A9 | A | ERROR | Methodology is not reproducible from the text. The ridge penalties (α = 50 for M2, 200 for M3, fixed by hand and never tuned) are not stated, and nothing says how hyper-parameters were chosen, which the exam requires. Only one scheme (expanding) is used, with no static or rolling comparison. EW turnover is reported as 0.000 because weight drift is ignored; this is not disclosed. | §3 vs code `Ridge(alpha=50.0)`, `Ridge(alpha=200.0)` |
| A10 | A | WEAKNESS | Missing required content: no "what you tried that did not work", no "what you would do next", no uncertainty (SE or interval) on any R² or Sharpe. The write-up refers to no figure, although four exist. Its "Table 1 / Table 2" numbering collides with notebook Tables 1/2, which are different tables. The literature review is a list of names with no stated expectation. | §§1, 4–7 |
| A11 | A | WEAKNESS | Unprofessional and irrelevant content: a heading "The Grader Defense Against Overfitting" and the claim that the pipeline "contains zero look-ahead and will execute reliably on any unseen evaluation dataset". §6.1 blames "2003–2010 macro coefficients" although the design refits annually. | §6.1–6.2 |
| A12 | A | WEAKNESS | The conclusion does not summarise the results. It gives no numbers and tells a PM to "rely on asset-level characteristics", the opposite of what Table 5 shows. "Cuts portfolio drawdowns in half" credits volatility-scaled targets for what is really a risk-parity effect: RP alone is −11.46% vs EW −24.15%. | §7 |
| A13 | A | STYLE | Lecture citation "Lecture 3 (p. 50)" for unpenalised intercepts: L3 p.50 is the orange-juice elasticity slide. | lectures/Lecture_3.txt p.50 |
| B1 | B | WEAKNESS | Too long and dense for its message (about 6,100 words). Internal process vocabulary leaks into the paper: "ledger", "added after review", "development scripts", "version control shows them only in a commit made after the development ledger had run", "AI guide §4a/§4b", "unit-tested". A referee would cut about a third. | §3, §4 |
| B2 | B | WEAKNESS | The title's "pre-registered" is stronger than B can show. B itself says the design is visible in version control only after the ledger ran, and that post-hoc diagnostics re-use the OOS window. Both are disclosed, but the label is still oversold. | Title; §3 "Design and timing" |
| B3 | B | WEAKNESS | B excludes KNN and neural networks as "outside the scope rule" because the course "does not teach" them, and gives similar reasons for covariance weights and costs. The exam explicitly allows "KNN … neural networks, or anything else". This is a self-imposed and mis-stated constraint. Cross-country macro transforms, which the exam suggests, were not run (acknowledged). | §5 "Considered and not used" |
| B4 | B | STYLE | The conclusion's "Adding global and country macro lowers out-of-sample R² in every paired comparison" refers to the 14 pairs of Table 3.18. The post-hoc interaction specification has a +0.012% point estimate, so it needs a qualifier (the abstract has one). | §6 vs Table B |
| B5 | B | STYLE | The conclusion is readable by a PM and gives numbers with uncertainty, but it uses "α over the three rules", "t = 1.74" and "volatility-scaled return" without a gloss. | §6 |
| B6 | B | (strength) | Provenance is clean: a headline table (3.31) prints every quoted number. All referenced tables and figures exist. There are no control characters or unbalanced `$`. Uncertainty is quantified everywhere (month bootstrap, a block-bootstrap check, power analysis, Bonferroni). Claims are calibrated ("not detectable rather than shown to be zero"). The P1 advantage is correctly traced to class-mean re-estimation (weight correlation 0.9994; the tilt regressor absorbs α), and fragility is shown (six-month dependence, 2011–17 vs 2018–24 split, 22% in asset_16). Asset identities are explicitly labelled hypotheses. Literature (Asness–Moskowitz–Pedersen 2013; Moskowitz–Ooi–Pedersen 2012; Gu–Kelly–Xiu 2020) is attributed correctly, and the L5 p.57, L8 p.62 and L1 p.33 citations check out. | cells 104, 106 |

---

## Scores (write-up only, out of 10)

**Solution A: 2.5 / 10.** The structure has the six sections, and the portfolio and attribution tables match the printed output. But:
- The Q2 evidence is fabricated: placebo numbers that contradict the printout, from a placebo that never ran.
- The Q1 conclusion contradicts A's own tables.
- The data summary contradicts A's own printed Table 1.
- Over 25 numbers have no printed source.
- Real-world identities are asserted as fact.
- The LaTeX is so badly damaged that variable names and leading digits are missing throughout.

**Solution B: 8.5 / 10.** This is a model negative-result paper. Every number traces to a printed cell, conclusions are calibrated to their SEs, and fragility is examined honestly. It loses points for length and process jargon, the oversold "pre-registered" framing, and the mis-stated "scope rule" used to exclude methods the exam allows.
