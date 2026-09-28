# R2: Blind review of Problem 2 (2.1 to 2.4), Solutions A and B

Both solutions use SEED = 2694. A uses the legacy `np.random.seed` API. B uses `SeedSequence(SEED).spawn(13)` named streams. The APIs differ, so the draws differ, and that is not penalised. I re-ran the key cells of both solutions outside the notebooks (scripts are in `reports/p2_work/`: `repro_A.py`, `repro_B.py`, `drift.py`, `anchored.py`). Every printed number I checked reproduced to the digit: A's one desk 0.096634, mean 0.7811, max 159.2464, top-10 share 0.6804, anchored median 0.999175. B's one desk 1.405e-05, mean 2.344, max 1566, top-10 share 0.8879, night-1 scenario kurtosis -0.2207. The dj30 numbers also reproduce: sd 0.011951, excess kurtosis 24.07, empirical VaR 3.325%, pipeline VaR 2.780%, 1,511 days.

## Scores

| | 2.1 (2) | 2.2 (6) | 2.3 (6) | 2.4 (6) | **Total (20)** |
|---|---|---|---|---|---|
| Solution A | 2.0 | 4.5 | 2.5 | 3.5 | **12.5** |
| Solution B | 2.0 | 6.0 | 5.5 | 5.5 | **19.0** |

---

## 2.1 Pre-registered prediction

- **A (cell 23).** Three sentences. It says no: log-variance drifts at about -1/n per night by Jensen's inequality, so the median is about e^-5 = 0.0067 and VaR is about 0.19 on night 2,500. The prediction is clear, reasoned and quantitative. 2/2.
- **B (cell 28).** Three sentences. The expectation (the "desk average") stays near 1. The median and the individual desk drift down at a rate proportional to -1/(n-1), so VaR collapses. The prediction is clear and reasoned, and less quantitative about night 2,500 than A's. 2/2.

The rubric also asks whether 2.3 reconciles against the 2.1 prediction. B does this explicitly, point by point (cell 38, "Reconciling with my 2.1 prediction"), and even marks its own claim "desk average approximately 1.0" as "only half right". A never refers back to its 2.1 answer (cell 29). That omission is scored under 2.3.

## 2.2 One desk, then a thousand

**Pipeline fidelity.** Both draw from N(0, sigma2-hat) with the SD as the scale and refit with `var(ddof=1)`. A's vectorised version multiplies by `var` of a standard-normal matrix. Because the step is scale-free, this is exactly equivalent.

**Night count.** Night 1 is the real library, whose variance is 1, so night 2,500 comes 2,499 refits later.
- **B** handles this correctly: `s2_one[0]=1`, a loop over `range(1, NIGHTS)`, and `logS2[0]` is night 1. It says so in cell 32.
- **A** starts at 1 and then applies 2,500 refits (`range(1, NIGHTS+1)`), so what it reports as "night 2,500" is really night 2,501. This affects the one desk, the 1,000 desks, and 2.4(a). I re-ran A with 2,499 steps: the median would be 0.007313 instead of 0.007512 and the one desk 0.096307 instead of 0.096634. The numerical effect is small, but the rules define night 1 explicitly.

**Numbers against printed output.**
- **B.** Every number in cell 32 is printed by cells 30 and 31: 1.405e-05, 0.0087 sigma, 0.010% of NAV, 2.344 / 0.005851 / 0.000029 / 1.225 / 1566, the VaR row 0.775 / 0.178 / 0.0125 / 2.575 / 92.1, 57.8% (MC SE 1.6%), 3.2%, 0.0073, 52.7% with rounding band 51.5 to 53.3%, and 0.214%.
- **A.** Several quoted numbers do not come from A's printed output. "Fallen by 69%" is not printed. "Over 90% collapsed below the starting value" is not printed; it is true (my re-run gives 93.9%), but no cell shows it. "95th percentile 1.284432" appears only as 1.2844. All the other numbers match cell 25.

**Is the story typical?** The fall from 2.8% to 0.24% implies sigma2-hat = (0.24/2.8)^2 = 0.00735.
- **B** computes this, places it at the 52.7th percentile, and concludes the story is the median outcome. That is the right analysis.
- **A** states that the story is "completely typical", but no cell computes the story's implied sigma2-hat or its percentile. In my re-run it sits at the 49.8th percentile of A's run, so the claim is right but unsupported. A's own desk ends at a VaR of 0.72 (0.87% of NAV on the story's scale), which is nowhere near the story's figure, so the reader cannot check the claim from what A shows.

**Where the one desk sits.** A reports the 79.3th percentile and B the 3.2th. Both are computed correctly.

## 2.3 Unbiased every night, wrong in the end

**Martingale check.**
- **A** runs 100,000 desks from sigma2 = 1 and gets a mean of 1.000002. This is fine.
- **B** runs 200,000 desks from two starts: sigma2 = 1 and the one desk's 1.4e-5. It fixed a CI rule in advance and reports honestly that the first start's CI (0.99935 to 0.99991, t = -2.62, p = 0.0088) misses 1. It computes the chance of such a miss (1.7%), states that the draws were not repeated, and cites the exact unbiasedness (L2 p.56). This is exemplary reporting, not an error. It also shows that the step is scale-free.

**Drift measurement.** This is the main difference between the solutions.
- **A's design.** A simulates one step from sigma2 = 1 with 5,000 desks. It does not use "a few hundred desks and a few hundred nights", so it has only 5,000 increments per n. The Monte Carlo SE is 0.00289 at n = 50, 0.00090 at n = 500 and 0.00028 at n = 5,000. At n = 500 that is 45% of the drift, and at n = 5,000 it is 141% of the drift. At n = 5,000 the design cannot even tell the drift from zero (my `drift.py`).
- **A's printed values** are -0.018645, **-0.003788** and -0.000233.
- **A's markdown** reports the "empirical change" as **-0.0204**, **-0.0020** and **-0.0002**. These are the formula values, not the measured ones. At n = 500 the printed measurement is almost twice the formula and 1.97 SE from the exact value -0.002005. Yet the markdown says it "matches ... within Monte Carlo sampling error" and presents the formula number as the measurement. Replacing printed results with theory values breaks rule 2 and is dishonest reporting. The formula is asserted from theory; A's own measurements do not establish it. A also calls -1/(n-1) "the exact theoretical formula". It is only an approximation: the exact value is digamma(k/2) - log(k/2), which is -0.020547 at n = 50.
- **B's design.** B uses 500 desks by 500 nights for each n, as the exam specifies. It reports the per-desk average nightly change with a correct SE: -0.019788 (0.000436), -0.002003 (0.000132) and -0.000173 (0.000040). n times the change is about -1 at all three sizes. A precise run at n = 50 (2,000 by 2,000) separates the candidates: it is 4.87 SE from -1/n and 0.90 SE from -1/(n-1). A cross-check on the 2.2 run gives -0.002059 (0.000042). A clearly labelled supplementary digamma cell confirms the theory. Every quoted value matches cell 35 or 37. The conclusion "-1/(n-1), which is -1/n for realistic n" is correct.
- **My independent run** (300 desks by 300 nights, `drift.py`) gives n times the drift of -0.975, -0.907 and -1.030 at n = 50, 500 and 5,000. This confirms that B's design, and not A's, is the one that establishes the formula.

**Histogram.** Both plot it. A marks the sample mean of log sigma2 and the theoretical -5. B overlays a fitted normal, the log of the mean, the log of the median and the one-tenth-VaR line, and adds a fan chart over nights.

**The three numbers.** Rubric item 3 asks for a reconciliation of expectation, median and sample mean.
- **A (cell 29)** gives: median e^mu, mean e^(mu + s^2/2) = e^0 under the log-normal approximation with mu = -5 and s^2 = 10, and a sample mean of 0.78 below 1 because a sample of 1,000 under-represents the right tail, where the top 10 hold 68.04% (printed). The substance is correct. The phrase "infinite right tail" is loose. The line "a single firm ... is mathematically guaranteed to collapse" overclaims: A's own output shows the 95th percentile at 1.28 and a maximum of 159 on night 2,500. Collapse happens almost surely only in the T to infinity limit.
- **B (cell 38)** gives the reconciliation in one paragraph: the nightly factor has mean 1 and median below 1, giving a log random walk with drift -1/n. B's predicted median, exp(-2,499 x 0.00206) = 0.0058, matches the measured 0.0059. The predicted sd is 3.17 against an observed 3.29, and the log cross-section is roughly normal. The top 10 desks hold 88.8% and the single largest holds 66.8%. Without that desk the mean is 0.78, and with it 2.34, which explains why the sample mean is "neither". B then explains why unbiasedness does not protect the pipeline: each night's error becomes tomorrow's truth, and no new information enters after night 1 (L2 p.57, p.71 to 72). All numbers are printed in Table 2.3c. This is the stronger answer.

**Format.** 2.3 asks for one paragraph.
- A answers with four numbered bullets and no paragraph.
- B has one proper reconciliation paragraph, but surrounds it with about 840 words of tables and bullets, well beyond what was asked.

## 2.4 What synthetic data can and cannot carry

**(a) Anchored library.**
- **The rule.** The Rules say every desk starts on night 1 from a library whose sample variance is exactly 1.
- **B** rescales each desk's 500 N(0,1) real days to a sample variance of exactly 1. Night 1 fits the real days alone, and then each night the library is the real days plus that night's 500 scenarios. B gets a median of 0.9955 and a band of 0.9406 to 1.059. It adds a demeaned robustness run and a stability table showing the band is in place by night 3.
- **A** uses raw N(0,1) real days, whose sample variance is not 1, but hard-codes the night-1 fit to 1. The library and its first fit are therefore inconsistent. A also runs 2,500 rather than 2,499 steps. It gets a median of 0.999175 and a band of 0.893898 to 1.120918.
- **My re-run (`anchored.py`).** With rescaled real days the band is 0.941 to 1.067, and sigma2-hat is uncorrelated with the real-day variance (correlation 0.00). With raw real days it is 0.893 to 1.129, and the correlation with each desk's real-day sample variance is 0.85 (0.86 in A's exact code). The real-day variance alone spans 0.898 to 1.103 across desks.
- **What this means for A.** Most of A's band is sampling variation in the real days, not noise accumulated from synthetic data. A then reads this band as "accumulation of noise" without making that distinction. Forcing night 1 to 1 makes no difference by night 2,500, because the chain forgets its start quickly.

**(b) dj30.**
- Both reduce the panel to 1,511 dates and get sd 0.011951, excess kurtosis 24.07, empirical 1% VaR 3.325% and pipeline VaR 2.780%. Both use all 1,511 days as the night-1 library; B says so explicitly.
- Night-1 scenario kurtosis is 0.398 for A and -0.221 for B. Both are ordinary values for a normal sample of 500.
- **B** adds a null distribution for the kurtosis (95% range -0.379 to 0.462, maximum 1.51 in 10,000 samples), a bootstrap SE on the VaR gap (0.54 pp, SE 0.32 pp, 1.7 SE), and it honestly calls that gap "suggestive rather than decisive". It also caveats that iid resampling ignores volatility clustering, and adds an exceedance rate of 1.65% against the model's 1%.
- **A** says the pipeline is "underestimating tail risk by nearly 20%". That figure is not printed: 3.33/2.78 - 1 = 19.6%, while (3.33 - 2.78)/3.33 = 16.4%.

**The paragraph.**
- **Classification.** Both classify correctly: (b) shows a loss of information, since the night-1 Gaussian draw already discards the fat tails; (a) shows an accumulation of noise that the real anchor bounds.
- **What synthetic data contains and lacks.** B's answer is the correct and complete one: the synthetic sample contains the fitted model, with its estimation error now treated as truth, plus fresh random noise. It lacks any new information and everything the model discards (tails, the March 2020 cluster, the mean). A says synthetic data "provides infinite volume and Gaussian tractability", which misses that what it adds is noise. A does mention "amplifies sampling noise" later.
- **What stops it.** Both say real data. B adds that it must be kept in every generation (even the same 500 days suffice, as (a) shows), that provenance tracking is required, and that real data cannot restore what the model class discards: getting the tails back needs a better model. A's "fresh, unpolluted empirical data" is slightly off, since (a) shows the same old real days are enough. A's "pins the variance to the true data-generating process" is also imprecise: the anchor pins it to the real sample's variance, which is exactly why A's band is so wide.

**Format.**
- A gives four numbered bullets, not one paragraph.
- B gives one paragraph of about 400 words, preceded by about 330 words of bullets. It is too long, but it does contain a single paragraph that answers every question asked.

---

## Findings

| id | solution | sub-q | severity | finding | evidence |
|---|---|---|---|---|---|
| A1 | A | 2.3 | FATAL | The markdown reports the measured drifts as -0.0204, -0.0020 and -0.0002. The cell prints -0.018645, -0.003788 and -0.000233, so the formula values were substituted for the measurements. At n = 500 the printed measurement is almost twice the formula, yet the markdown says it matches "within Monte Carlo sampling error". | Cell 28 output vs cell 29 point 2 |
| A2 | A | 2.3 | ERROR | The drift design is a single step with 5,000 desks, not "a few hundred desks and a few hundred nights". The MC SE is 45% of the drift at n = 500 and 141% at n = 5,000, so the measurement cannot establish the formula, and at n = 5,000 cannot even tell the drift from zero. The formula is asserted, not measured. | Cell 28; my `drift.py` (SE 0.00090 and 0.00028) |
| A3 | A | 2.2/2.3/2.4a | ERROR | Off by one: 2,500 refits after night 1, so the reported "night 2,500" is night 2,501. | Cell 25 `range(1, NIGHTS+1)`, `sigma2_one[0]=1`; cell 31 `range(NIGHTS)` |
| A4 | A | 2.4a | ERROR | The real days are raw N(0,1) draws whose sample variance is not 1, yet the night-1 fit is forced to 1, which is inconsistent with the Rules. The reported band of 0.894 to 1.121 is mostly real-sample variation (correlation 0.86 with each desk's real-day variance), not accumulated synthetic noise. With rescaled real days the band is about 0.941 to 1.067. | Cell 31; my `anchored.py` |
| A5 | A | 2.3 | ERROR | 2.3 never reconciles with the 2.1 prediction, which 2.1 says it must. | Cell 29 |
| A6 | A | 2.2 | ERROR | Numbers not printed by any cell: "69%", "over 90% collapsed below the starting value" (true value 93.9%, not shown), and 1.284432 (printed as 1.2844). | Cell 26 vs cell 25 output |
| A7 | A | 2.2 | WEAKNESS | "The story is completely typical" is asserted without computing the story's implied sigma2-hat of 0.0073 or its percentile. The claim is true (49.8th percentile in my re-run of A), but it is unsupported, and A's own desk ends at a VaR of 0.72. | Cells 25 and 26 |
| A8 | A | 2.4b | ERROR (minor) | "Underestimating tail risk by nearly 20%" is not printed, and the base is ambiguous (19.6% or 16.4%). | Cell 32 vs cell 31 |
| A9 | A | 2.3/2.4 | WEAKNESS | Both answers are four numbered bullets, not the required single paragraph. | Cells 29 and 32 |
| A10 | A | 2.3 | WEAKNESS | Overclaim: a single path "is mathematically guaranteed to collapse", while A's own output shows a 95th percentile of 1.28 and a maximum of 159 on night 2,500. -1/(n-1) is also called "exact"; it is an approximation (exact value -0.020547 at n = 50). | Cells 29 and 25 |
| A11 | A | 2.4 | WEAKNESS | "Contains" is answered as "infinite volume and Gaussian tractability", missing that what synthetic data adds is noise and estimation error treated as truth. "Pins the variance to the true DGP" should be "to the real sample's variance". "Fresh" data is not needed; (a) shows the same real days suffice. | Cell 32 |
| A12 | A | 2.1 | none | Clear quantitative three-sentence prediction. | Cell 23 |
| B1 | B | 2.3 | STYLE | The CI miss at sigma2 = 1 (p = 0.0088) is reported honestly, with its probability and a no-rerun statement. This is noted as good practice, not a defect. | Cell 34 |
| B2 | B | 2.3/2.4 | WEAKNESS | Very long answers (about 840 and 740 words). Each contains one required paragraph, but the answer as a whole far exceeds "one paragraph". | Cells 38 and 43 |
| B3 | B | 2.1 | STYLE | The prediction for night 2,500 is qualitative ("collapse", "severe underestimation") with no magnitude. Acceptable. | Cell 28 |
| B4 | B | 2.4b | STYLE | The night-1 library is all 1,511 days rather than the manual's 500. B discloses this explicitly (A does the same silently). | Cell 43 |
| B5 | B | all | none | Every quoted number checked against printed output matched. The night count is correct (2,499 refits). The drift design follows the spec, with SEs, and discriminates -1/n from -1/(n-1). The real days are rescaled to variance exactly 1. Lecture citations (L2 p.56, 57, 71 to 72, 80) are apt. | Cells 30 to 43; reproduced in `repro_B.py` |

## Bottom line

**A** has the right storyline and correct simulation mechanics. But its 2.3 drift "measurements" are formula values presented as data. Its drift design is too small to support the formula. It gets the night count wrong. It does not rescale the anchored real days, and then misreads the resulting band. It quotes several unprinted numbers, never reconciles 2.3 with its 2.1 prediction, and answers in bullet lists where one paragraph was required.

**B** follows every rule. It reports every number from a cell, measures and discriminates the drift formula with proper Monte Carlo error, reconciles against 2.1 point by point, and gives the most complete conceptual answer in 2.4. Its only real fault is verbosity.
