# Holdout check (dry run)

- training 2003-01 to 2022-12-31 (models frozen there); holdout 2023-01-31 to 2024-12-31, 24 months, 50 assets ({'A': 26, 'B': 8, 'C': 9, 'D': 7} by class)
- comparison: 2011-01 to 2022-12-31 on the same assets, expanding refits each December
- correctness gate: the rebuilt pipeline reproduces the notebook's ridge M2 result, 0.042% (notebook 0.042%)
- penalties chosen (alpha/n) at 2022-12-31: M2r ridge on x2, x5: 1e+03, M2n ridge M2, missing ranks neutral: 1e+03

## R2_OOS (holdout)

|                                                                 |   R2_OOS |     SE |
|:----------------------------------------------------------------|---------:|-------:|
| ('M1 class means', 'pooled trailing mean')                      |   0.591% | 0.328% |
| ('M1 class means', 'zero')                                      |   0.353% | 1.271% |
| ('M2r ridge on x2, x5', 'pooled trailing mean')                 |   0.591% | 0.328% |
| ('M2r ridge on x2, x5', 'zero')                                 |   0.353% | 1.271% |
| ('M2n ridge M2, missing ranks neutral', 'pooled trailing mean') |   0.591% | 0.328% |
| ('M2n ridge M2, missing ranks neutral', 'zero')                 |   0.353% | 1.271% |

## R2_OOS by class, holdout, vs the pooled trailing mean

|                                     |      A |      B |      C |       D |
|:------------------------------------|-------:|-------:|-------:|--------:|
| M1 class means                      | 0.047% | 3.551% | 2.318% | -1.645% |
| M2r ridge on x2, x5                 | 0.047% | 3.551% | 2.317% | -1.645% |
| M2n ridge M2, missing ranks neutral | 0.047% | 3.551% | 2.317% | -1.645% |

## R2_OOS (comparison period, same assets)

|                                                                 |   R2_OOS |     SE |
|:----------------------------------------------------------------|---------:|-------:|
| ('M1 class means', 'pooled trailing mean')                      |   0.016% | 0.269% |
| ('M1 class means', 'zero')                                      |  -0.317% | 0.613% |
| ('M2r ridge on x2, x5', 'pooled trailing mean')                 |  -0.029% | 0.275% |
| ('M2r ridge on x2, x5', 'zero')                                 |  -0.361% | 0.626% |
| ('M2n ridge M2, missing ranks neutral', 'pooled trailing mean') |  -0.035% | 0.274% |
| ('M2n ridge M2, missing ranks neutral', 'zero')                 |  -0.368% | 0.620% |

## Portfolios, net of 10 bp (holdout)

|          |   net Sharpe |    SE |   mean net return (annual) |
|:---------|-------------:|------:|---------------------------:|
| EW       |        0.316 | 0.756 |                      0.023 |
| RP       |        0.225 | 0.759 |                      0.012 |
| TSMOM    |       -0.196 | 0.740 |                     -0.006 |
| P1 (M1)  |        0.680 | 0.773 |                      0.030 |
| P1 (M2r) |        0.680 | 0.773 |                      0.030 |
| P1 (M2n) |        0.680 | 0.773 |                      0.030 |

- P1 (M2r) minus RP, net Sharpe (SE): +0.45 (0.19)
- avg monthly weight correlation, P1 (M2r) vs P1 (M1): 1.0000
- P1 (M2r) share of gross in class D: 51%

## Portfolios, net of 10 bp (comparison period, same assets)

|          |   net Sharpe |    SE |   mean net return (annual) |
|:---------|-------------:|------:|---------------------------:|
| EW       |        0.271 | 0.298 |                      0.025 |
| RP       |        0.311 | 0.297 |                      0.017 |
| TSMOM    |        0.160 | 0.291 |                      0.007 |
| P1 (M1)  |        0.394 | 0.301 |                      0.017 |
| P1 (M2r) |        0.402 | 0.301 |                      0.016 |
| P1 (M2n) |        0.399 | 0.301 |                      0.016 |

- P1 (M2r) minus RP, net Sharpe (SE): +0.09 (0.13)
- avg monthly weight correlation, P1 (M2r) vs P1 (M1): 0.9953
- P1 (M2r) share of gross in class D: 61%
