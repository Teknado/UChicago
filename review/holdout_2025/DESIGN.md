# Holdout check on real 2025–26 data: design, fixed before any 2025–26 number is seen

*Branch `financial_analytics_review` only. This is an informal robustness check requested by the student. It is **not**
part of the submission, and nothing in it feeds back into the submitted notebook. It is post hoc by nature: the models
were designed and evaluated on 2000–2024 before it was run. Written 2026-09-29, before any data after 2024-12 was
downloaded.*

## Question
If the frozen, end-2024 versions of our Problem 3 approaches are applied to real returns from 2025-01 onwards, do they
behave as the 2011–2024 evaluation says they should? In particular:
- no detectable forecasting skill beyond class means;
- a P1 portfolio that is essentially the class-means tilt.

The holdout is short (about 20 months). The test can reveal a break or a gross error, but it cannot confirm or overturn
the notebook's conclusions: a Sharpe ratio over 20 months has a standard error of roughly 0.8 (annualised).

## 1. Data (downloaded only when network access allows)
| what | source | series |
|---|---|---|
| equity indices (class C, 9) | Yahoo Finance, monthly | price indices `^GSPC ^N225 ^SSMI ^GDAXI ^FTSE ^GSPTSE ^FCHI FTSEMIB.MI` and both `^AEX` and `^IBEX` for asset_42; USD ETFs with dividends (adjusted close) `SPY EWJ EWL EWG EWU EWC EWQ EWI EWN EWP` |
| currencies (class B, 8) | Yahoo Finance, monthly | `JPY=X CHF=X EURUSD=X GBPUSD=X CAD=X SEK=X AUDUSD=X NZDUSD=X` |
| government bonds (class D, 7) | FRED, monthly | 10-year yields `IRLTLT01{US,JP,CH,DE,GB,CA,SE}M156N`; Yahoo `ZN=F` (US 10-year note futures) |
| commodities (class A, 26) | Yahoo Finance, monthly | front-month continuous futures `CL=F BZ=F NG=F HO=F RB=F GC=F SI=F PL=F PA=F HG=F ALI=F ZC=F ZS=F ZW=F KE=F ZL=F ZM=F ZO=F KC=F SB=F CT=F CC=F OJ=F LE=F HE=F GF=F` |
| short rates | FRED, monthly | `TB3MS` (US 3-month bill); 3-month interbank `IR3TIB01{US,JP,CH,EZ,GB,CA,SE,AU,NZ}M156N` |

## 2. Rebuilding the panel's returns, and the acceptance rule
The panel's exact return definitions are not documented, so each asset gets a fixed, short list of candidate
constructions. The candidate that best matches the panel on **2019-01 to 2024-12** (72 months, all before the holdout) is kept:
- **Equities:** (i) local price return minus the US bill rate; (ii) local price return minus the local short rate; (iii) USD ETF total return minus the US bill rate.
- **Currencies:** (i) the spot return of the foreign currency against USD; (ii) spot return plus the interest differential (foreign minus US 3-month rate)/12, i.e. the carry-inclusive excess return.
- **Bonds:** a duration approximation from the 10-year yield, r ≈ y_{t−1}/12 − D·Δy − (local short rate)/12, with D the modified duration of a 10-year par bond at y_{t−1}; for the US also (ii) the `ZN=F` price return.
- **Commodities:** the front-month futures price return (already an excess return). The three named commodities are asset_33 WTI, asset_15 Brent and asset_6 gold. Every other class-A asset is matched to the candidate contract with the highest 2019–2024 correlation, and each contract is used at most once.

**Acceptance: correlation with the panel ≥ 0.95 over 2019–2024.** The threshold was fixed here, before any holdout data. An asset that fails is dropped from the holdout; it is not repaired. The match table (correlation, RMSE, mean difference) is reported in full.

## 3. The holdout panel
The panel runs 2000–2024 as given, followed by the rebuilt returns for accepted assets from 2025-01 to the last complete
month (at least 12 months are required, or the test is not run).
- **End month:** the last month for which at least 90% of the accepted assets have a rebuilt return. An accepted asset
  with any missing month between 2025-01 and that end month is dropped (no filling of returns).
- **Short rates** (FRED publishes with a lag) are carried forward by at most 3 months. A return that would need an
  older rate is missing.

*These two rules were added while writing `rebuild.py`, still before any 2025–26 data was downloaded.*
- σ̂ (36-month SD of past excess returns) and `x2` (compounded 12-month return, t−12..t−1) are computed from the spliced
  series, exactly as defined in the notebook.
- `x1`, `x3` and `x4` cannot be rebuilt for 2025+, so they are not used.

## 4. Models (all fitted on 2003-01 to 2024-12 and then frozen; nothing is refitted on holdout data)
| model | what it is | why |
|---|---|---|
| **M1: class means** | the class means of y (the notebook's M1) | the model the notebook finds P1 is equivalent to |
| **M2r: ridge on the rebuildable characteristics** | the notebook's `LinearFE` ridge with unpenalised class intercepts on the within-class ranks of `x2` and `x5`; α/n tuned on the notebook's three annual forward folds inside 2003–2024 | the closest honest version of the primary model when `x1`, `x3`, `x4` are unavailable |
| M2n (secondary): full ridge M2, neutral fill | the notebook's M2 fitted on 2003–2024 with all five ranks; in the holdout the three unavailable ranks are set to 0 (the class average) | shows how much the missing characteristics would have mattered |

**Forecast benchmarks:** the pooled trailing mean of y, updated monthly through t−1 over the spliced panel (primary, as in the notebook), and zero.

**Ranks.** Training uses the panel's own ranks. Holdout ranks are taken within class among the accepted assets.

## 5. What is reported (no decision rule: this is descriptive)
- R²_OOS of M1, M2r and M2n against the pooled trailing mean and against zero, pooled over accepted assets and by class, with a month-bootstrap SE (B = 10,000; whole months resampled, forecasts held fixed).
- Portfolios on the accepted assets, as in the notebook (unit gross, monthly rebalancing, 10 bp per unit of Σ|Δw| on drifted weights):
  - EW, RP ∝ 1/σ̂ and TSMOM ∝ sign(R12)/σ̂;
  - P1(M1), P1(M2r) and P1(M2n).

  Reported: net Sharpe ratios with a month-bootstrap SE, P1 minus RP, the weight correlation of P1(M2r) with P1(M1), and the class-D share of P1.
- The same statistics for 2011–2024 on the same accepted assets, for comparison.

**What we expect, stated in advance:**
- R²_OOS near zero for every model, with SEs larger than any plausible signal;
- P1(M2r) weights almost identical to P1(M1);
- P1 concentrated in bonds;
- Sharpe differences far inside their SEs.

A result outside this is a reason to look for an error first, and only then for a change in the data.

## 6. Dry run (to test the code, not the models)
Before any real data arrives, the whole harness runs on the existing panel with 2023-01 to 2024-12 standing in as the
"holdout". In the dry run the models are fitted on 2003–2022, the panel's own returns stand in as the "rebuilt" series
(so every asset is accepted), and only `x2` and `x5` are used. The dry run checks the code path only; its numbers are not
a result.

## 7. How to run (from the repository root)
```
python3 review/holdout_2025/fetch.py              # downloads into raw/ (needs Yahoo Finance and FRED access)
python3 review/holdout_2025/rebuild.py            # match table and extended returns into data/
python3 review/holdout_2025/holdout.py real       # results into results_real/
```
`rebuild.py --selftest` checks the matching code on synthetic downloads made from the panel. `holdout.py dry` is the
dry run of §6. Before it reports anything, `holdout.py` reproduces the notebook's ridge M2 expanding R²_OOS (0.042%)
and stops if the value differs.
