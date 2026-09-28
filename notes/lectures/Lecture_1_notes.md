# Lecture 1 — Introduction, Data Visualization, and Dimension Reduction
BUSN 41210 Financial Analytics, Chicago Booth, Autumn 2026 (Dacheng Xiu). Lecture date on the title slide: September 8, 2026.

Source: `/home/user/UChicago/Lecture_1.pdf` (82 PDF pages). **Citation convention: "L1 p.X" = PDF page X.**
I read the whole extracted text, then rendered and looked at every one of the 82 pages (as 2x3 contact sheets at 110 dpi, plus
p.36, 38, 39, 42, 43, 44 at 200 dpi). Formulas and figures below are transcribed from the rendered slides.

> **Bottom line for the exam boundary.** L1 teaches one method in full: **PCA via the eigen-decomposition of the sample
> covariance matrix** (L1 p.45–74). It also defines **covariance and correlation** and the **correlation matrix /
> heat map via `df.corr()`** (p.44), and uses the **SVD biplot** and the **scree plot** as visual tools (p.42–43).
> Everything else is motivation: why finance is hard (low SNR, small n, nonstationarity, n ≈ p; p.33–34), data
> visualization principles (p.37–41), and the course philosophy on AI tools: framing, validation, judgment, benchmarks,
> and not letting the future into the training set (p.80–82). No sklearn/statsmodels code appears in L1. The only
> code shown is `df.corr()`. L1 has no R²_OOS, no cross-validation, no train/test splitting procedure, and no
> regression or classification estimators. Those come in later lectures.

---

## 0. PDF page to printed frame number map
Beamer overlays make PDF pages and the printed frame number differ. Always cite the PDF page.

| PDF page(s) | printed # | PDF page(s) | printed # | PDF page(s) | printed # |
|---|---|---|---|---|---|
| 1–6 | 1–6 | 33 | 23 | 59–60 | 49 |
| 7–14 | 7 (build) | 34 | 24 | 61 | 50 |
| 15 | 8 | 35 | 25 | 62 | 51 |
| 16 | 9 | 36 | 26 | 63–74 | 52–63 |
| 17–18 | 10 | 37–46 | 27–36 | 75–79 | 64–68 |
| 19–28 | 11–20 | 47–58 | 37–48 | 80 | 69 |
| 29–30 | 21 | | | 81 | 70 |
| 31–32 | 22 | | | 82 | 71 |

---

## 1. Course framing and logistics (L1 p.1–4)

- **p.1** Title: "Lecture 1: Introduction, Data Visualization, and Dimension Reduction", Dacheng Xiu, Chicago Booth, Sept 8, 2026.
- **p.2 Course overview (9 lectures):**
  1. Introduction, Data Visualization, and Dimension Reduction
  2. Linear Regression Models
  3. Transformations, Interactions, Categorical Variables, Multiple Linear Regressions
  4. Model Selection Principles, Screening, Information Criteria, Stepwise Selection
  5. Regularization in Linear Models: Lasso & Ridge
  6. Classification, Multinomials, KNN, Sensitivity/Specificity
  7. Clustering, Mixture models, K-means
  8. Trees, Random Forests, Boosted Trees, Variable Importance
  9. Machine Learning in Finance: Advanced Applications
  - *Note:* the local folder has only Lecture_1 … Lecture_8 PDFs. Lecture 9's content is not available here, so anything that exists only in L9 is outside the verifiable boundary.
- **p.3 JupyterHub:** `https://jupyter-class.chicagobooth.edu/`, CNet ID login. Shared data folder is **`/classes/41210_MiF_fall2026/`**, and code runs on the Booth server. This matches the exam's `_DATA_DIR = '/classes/41210_MiF_fall2026/Data/'`. You don't need Python installed locally.
- **p.4 Outline:** (1) Introduction: What is Big Data? What is ML? Can machines learn finance? (2) Data Visualization. (3) Dimension Reduction.

## 2. "Big data" and ML successes (L1 p.5–14). Motivation only.
- **p.5** "Data: The New…" is a collage of headlines ("Data is the new gold", "Data as the new currency", "Data isn't 'the new oil' – it's way more valuable", WSJ "Data Is the New Currency", "Is data the new currency?").
- **p.6** Domo "Data Never Sleeps 5.0" infographic (2018, "every minute of the day": Google 3,877,140 searches, YouTube 4,333,560 videos, etc.).
- **p.7–14** is one frame built up in overlays, "Machine Learning Rocks! Machine Learning Successes": 1997 Deep Blue beats Kasparov, 2009 Google self-driving car, 2011 Watson wins Jeopardy!, 2012 Microsoft translates English to Chinese in real time, 2016 AlphaGo beats Lee Sedol, 2021 AlphaFold predicts protein structure, 2022 OpenAI's ChatGPT. Photos only.

## 3. What is Machine Learning? (L1 p.15–25)
- **p.15** A humorous "Machine learning." tweet photo (Schwarzenegger at the gym). No content.
- **p.16 Paradigm shift:** from machine *programming* to machine *learning*.
  - Conventional programming tells the computer what to do by breaking big problems into many small, precisely defined tasks.
  - ML **learns (estimates) from observational data**, instead of requiring pre-specified logic, for decision making and problem solving.
- **p.17–18 Email-validity example** ("Is this a valid email address? dacheng.xiu@chicagobooth.edu"):
  - Conventional: `IF` upper/lowercase letters or digits `AND` "@" followed by a valid domain `AND NOT` special characters "!#$…" `AND …` `THEN` valid `ELSE` invalid.
  - ML: (1) (Big) data on valid/invalid addresses. The toy table is Y/N | Address: 0 `jaime@lannister`; 1 `hound@clegane.com`; 0 `john.snow@GO†.edu`; …
    (2) **Stats model:** $Y/N = b_0 + b_1(\text{@}) + b_2(\text{!\#\$\&\%?*}) + \dots$ (a linear model in indicator features),
    (3) = **estimated probability of valid email**.
  - This is the first appearance of "features → model → probability". It is not developed here. Classification comes in L6.
- **p.19–23 MNIST "What Number is This?":** a 28px × 28px handwritten "3" (p.19). Each pixel becomes a value from 0 to 1 by grayscale brightness (p.20). How do you code an algorithm (a step-by-step procedure) that knows different 3s are the same? (p.21–22). "You need a flexible approach that can learn patterns of what makes a three a 'three', and that's where machine learning shines" (p.23). **Figure p.23: a neural network with 784 inputs, 2 hidden layers of 16 neurons each, and 10 output labels (0–9).** NN is shown pictorially only and is not taught.
- **p.24** Arthur Samuel coined "Machine Learning" in 1959 ("Some studies in machine learning using the game of checkers"). ML combines statistics (extracting information from data) with computational ideas (efficient implementation on large data). "Machine learning is to big data as human learning is to life experience." ML mimics human learning: catching a ball by trial and error, without first learning the motion equations.
- **p.25 ML paradigms (definitions the course uses):**
  - **Supervised learning ("predictive analytics"):** systems and algorithms that determine a predictive model using data points with known outcomes. **Regression** means the outputs are real variables. **Classification** means the model finds the classes in which to place its inputs. Examples: predict default, housing prices, asset returns.
  - **Unsupervised learning ("exploratory data analysis"):** learns patterns from unlabeled data. Examples: recommender systems (Amazon "frequently bought together", Netflix "Top Picks"), language modeling in NLP. *(PCA, taught later in L1, is placed here: p.46 calls it "the most common unsupervised learning approach to dimension reduction".)*
  - **Reinforcement learning:** an agent makes a sequence of decisions in an uncertain environment, using trial and error, rewards and penalties, and maximizes total reward. Examples: AlphaStar/Atari/StarCraft, AlphaGo, self-driving cars. There is a pointer to the AlphaGo documentary. **Mentioned only. Out of scope.**

## 4. Can machines learn finance? (L1 p.26–36)
- **p.26** Machines seem capable of anything (speech recognition, translation, driving, chess/Go…), "but can they learn finance?"
- **p.27 The Hype:** headlines. CNBC Jul 12 2017 "Machines taking over hedge funds despite lack of evidence they outperform humans". Economist Dec 2017 "Hedge funds embrace machine learning—up to a point". Economist May 2017 "Machine-learning promises to shake up large swathes of finance". Bloomberg Jul 15 2017 "The U.S. Stock Market Belongs to Bots". Bloomberg Aug 9 2017 "The Quant Fund Robot Takeover Has Been Postponed", annotated "Same Reporter, 3 weeks later".
- **p.28 ML adoption:** Morgan Stanley survey (FT chart) "Investment groups gradually embracing machine learning", 2016–2019. Stacked bars: not under consideration / under consideration but not used / one component of investment process / central to investment process / unknown. "Not under consideration" shrinks over time.
- **p.29–30 AIEQ ETF:** "first actively-managed ETF to utilize artificial intelligence throughout the investment process… models developed by Equbot with IBM Watson". p.30 adds "Performance?" with a Yahoo Finance chart of AIEQ vs a comparison line, 2020–2024. AIEQ (blue) ends at about +36.95% and the comparison line (teal, unlabeled on the slide) at about +84.6% over the window. **Lesson (implicit): AI branding does not guarantee beating a simple benchmark.**
- **p.31–32 Sentient Investment Management:** MIT Tech Review (Feb 4 2016) "Will AI-Powered Hedge Funds Outsmart the Market?". "Trading, Evolved": evolutionary intelligence, deep learning, distributed AI, "enormous stockpiles of data". p.32 adds Bloomberg Sept 6 2018 "AI Hedge Fund Is Said to **Liquidate** After Less Than Two Years."
- **p.33 "But Why? Finance is Different!"** *(key domain-knowledge slide)*
  - **Low signal-to-noise ratios (SNR)**, in sharp contrast with computer science.
  - Domain knowledge (thanks to Gene Fama): **market efficiency**. Returns *must* be dominated by *news* in well-functioning markets.
  - "Low SNRs are not a coincidence … market efficiency reinforces it!"
- **p.34 "Need for Machines"** *(key slide)*
  - "Big data" is not just about the size of the data!
  - Historical data in finance and economics are typically **not large: barely 10s of years, 100s of months. Lack of stationarity effectively shrinks the counts further.**
  - Finance does have lots of variables. **With n = 100 data points, it is infeasible to run linear regressions with p = 100 variables. In cases like this (n ≈ p), ML is inevitable.**
  - Why ML for finance? (i) **lots of explanatory variables with potentially high correlations**; (ii) **functional form is unknown and likely complex (nonlinear).**
- **p.35 "'ML' is Inevitable":** Has ML been used before? Yes, perhaps without knowing it. The Chicago Booth Review (Feb 05 2024) piece "In Finance, Humans Were the First Machines" lists:
  - **nonlinearity:** sorting returns by characteristics
  - **dimension reduction:** employing portfolios instead of individual assets
  - **variable selection and factor analysis:** e.g., Fama-French factors
  - **priors (regularization):** incorporating economic intuition/theory
  - "Why not embrace modern ML techniques?"
- **p.36 "ML-based Portfolios":** Gu, Kelly & Xiu (2019) "Empirical Asset Pricing via Machine Learning". Cumulative performance, 1987–2016, of long (solid) and short (dashed) legs of ML-forecast-sorted portfolios for **OLS-3+H, PLS, PCR, ENet+H, GLM+H, RF, GBRT+H, NN3**, with the **SP500−R_f** market excess return in black as the benchmark. Y-axis: "Long Position" from 0 up to about 8, "Short Position" from 0 down to 4. Grey bars mark recessions (about 1990–91, 2001, 2008–09). NN3 has the highest long leg (about 8.5) and the lowest short leg. All ML long legs beat SP500−Rf (about 2.3). **Legend labels only. The models are not explained in L1.** ("+H" is not defined on the slide.)

## 5. Data visualization (L1 p.37–44)
- **p.37 Guidelines:** data visualization is an essential part of **exploratory data analysis (EDA)**.
  - **Statistics:** reduce the dimension of your data to a few rich variables for comparison. This can be just picking two features to scatterplot, or can involve more complicated projections.
  - **Design:** effective communication, with shapes, space, and color, for a given set of variable observations.
  - **Language:** making it easy to move from Stats to Design.
  - "They're all interconnected, but **we'll focus on statistics**."
- **p.38 Example, time series:** two stacked panels for Apple, 2016-01-04 to about 2021-12 (x-axis ticks as `YYYYMMDD` strings, e.g. 20160104 … 20210727):
  - Top: **"Time series for Apple Stock Price"**. The price rises to about 500 and then drops to about 125 at a red dashed vertical line annotated with an arrow, "Apple announces 4-1 stock split" (Aug 2020).
  - Bottom: **"Time series for Apple Stock Return"**, roughly ±0.10, with the same red dashed line. **There is no −75% return at the split**, and the volatility cluster in early 2020 (COVID) is visible.
  - *Takeaway (my reading; the slide shows it rather than stating it):* unadjusted **price levels** carry artefacts such as splits, and they are non-stationary. **Returns** are the object to model. Always plot the raw series to catch data problems. Plot style: matplotlib/seaborn whitegrid, `axvline`-style dashed red event marker, and an arrow annotation.
- **p.39 "Would you like to invest in this new product (red)?"** Growth of 100, Dec 1990 to Dec 2007: **S&P 100 (yellow), Lehman Bond Index (orange), FS (blue), Simulated 3x FS (red)**. FS is an implausibly smooth, steadily rising line. The simulated 3× levered version climbs to about 2,000.
  - *Lesson (implicit):* a picture can make an implausible return stream look attractive. Be suspicious of too-smooth performance and of *simulated* (not realized) track records with leverage. *(Outside knowledge, not on the slide: "FS" in this well-known chart is the Fairfield Sentry fund, a Madoff feeder.)*
- **p.40 Spatial data:** map of Walmart stores, supercenters, and distribution centers (source: Kosuke Imai, "Quantitative Social Science").
- **p.41 Spatial-temporal data:** Walmart's expansion over 1975, 1985, 1995, 2005 (Imai).
- **p.42 Panel data, the biplot:**
  - "A **Biplot** is constructed by the **SVD** to obtain a low-rank approximation to the data
    $$X \approx Z = U D V^{\top},\quad X:\ T\times n,\ U:\ T\times 2,\ D:\ 2\times 2,\ V:\ n\times 2."$$
  - Two panels, Component 1 vs Component 2 loadings (rays from the origin) for S&P 100-type stocks using high-frequency data. Blue is "Non-Financial" and red is "Financial".
    Week 03/07/2005–03/11/2005: financials are mixed in with non-financials.
    Week 03/10/2008–03/14/2008 (the Bear Stearns week): **financials (AIG, C, GS, BAC, JPM, LEH) separate into their own direction** (negative Component 2).
  - Source: Aït-Sahalia & Xiu (2019), "Principal Component Analysis of High-Frequency Data", JASA 114(525), 287–303.
  - *Takeaway:* low-rank (PCA/SVD) projections reveal **block/sector structure** in a panel, and that structure can change over time (crisis).
- **p.43 Scree plot:** "A **scree plot** is a graph of eigenvalues against the corresponding PC number."
  - Left: about 90 eigenvalues. The first is about 2.45, the second about 0.4, then a fast decay toward 0. **One dominant factor.**
  - Right: time series, 2003–2013, of the 1st (blue), 2nd (black dashed), 3rd (red dotted) and "≥4 Average" (green) eigenvalues. The first rises sharply in 2008–2009 and 2011 (crisis co-movement). The others stay near 0.03–0.1.
  - Same source (Aït-Sahalia & Xiu 2019).
  - **No formal rule for choosing k is given in L1.** There is no variance-explained threshold, no elbow rule stated explicitly, and no CV of k. The scree plot is presented only as the visual tool.
- **p.44 "What 'Correlated' Means"** *(the only slide with a code idiom)*
  - "Two variables are correlated if knowing one tells you something about the other."
  - $$\mathrm{Cov}(X,Y)=\mathbb{E}\big[(X-\mathbb{E}X)(Y-\mathbb{E}Y)\big],\qquad \rho_{XY}=\frac{\mathrm{Cov}(X,Y)}{\mathrm{sd}(X)\,\mathrm{sd}(Y)}.$$
  - Intuition: when X is above *its* average, is Y above *its* average? Multiply the two deviations and average. The size depends on units, so divide by both SDs. **"correlation is covariance in units that cancel."**
  - ρ lies between −1 and +1: +1 is a perfect line sloping up, −1 sloping down, 0 **no *linear* relation**.
  - **A correlation matrix is every pair at once: diagonal 1, symmetric. A heat map of it is the fastest way to see structure in a wide dataset. `df.corr()`.**
  - **Warning:** "ρ = 0 does not mean 'unrelated', only no *straight-line* relation: a perfect U-shape has ρ ≈ 0. **Look at the scatter plot as well as the number.**"
  - Bridge to PCA: "columns that are highly correlated are not carrying that many separate pieces of information."

## 6. Dimension reduction and PCA (L1 p.45–74). **The method taught in L1.**

### 6.1 Setup (p.45–46)
- **p.45** $X$ is a dataset with **T observations of n variables**. **Each variable was previously demeaned.** Write
  $$X=(X_1,\dots,X_n),\qquad X_i=(X_{i1},\dots,X_{iT})',\ i=1,\dots,n.$$
  Goal: "Extract as much information (**variance**) of X and build another dataset Z with T observations but only **k ≪ n** variables": $Z=(Z_1,\dots,Z_k)$.
  - Orientation convention: rows are time, columns are variables. A PC $Z_j$ is a **T-vector** (a factor time series).
- **p.46** PCA is "the most common **unsupervised** learning approach to dimension reduction". Uses: visualization of large datasets; efficient use of resources (e.g., compression); **noise reduction** (improving data quality); **pre-treatment of the data for further ML algorithms**. It applies to any situation that requires dimension (number of features) reduction. It accomplishes the reduction by "**rotating**" the dataset in a smart way.

### 6.2 Geometric intuition, the camera (p.47–57)
- **p.47** You want to take a picture of a group of people standing in an elongated diagonal cloud.
- **p.48** Which camera position is best: I (left, looking right), II (bottom-left, diagonal), III (bottom, looking up), or IV (right, perpendicular to the long axis)?
- **p.49–50** Camera I: projection onto a vertical line. People are spread out but somewhat overlapping.
- **p.51–52** Camera II: its line is perpendicular to the long axis of the cloud, so everyone collapses into a clump (minimal spread).
- **p.53–54** Camera III: projection onto a horizontal line, moderate spread.
- **p.55–56** Camera IV: projection onto a line **parallel to the long axis**, maximal spread.
- **p.57** Comparing the cameras (I, II, III, IV projections side by side): "**Position of camera IV seems to be the best one.**" The best projection is the direction of **maximum variance** of the projected points.

### 6.3 Covariance as a linear map (p.58–62)
- **p.58** "How should we find the best position for the camera? X is a set of points in ℝ² with sample covariance $\widehat{\Sigma}$." Scatter of $(X_1,X_2)$, positively correlated.
- **p.59–60** "What does the transformation $Z = X\widehat{\Sigma}$ do?" The right panel plots $(Z_1,Z_2)$, stretched further along the diagonal.
- **p.61** "Circle → ellipse": the unit circle in X-space maps to a thin ellipse elongated along the main diagonal.
- **p.62** "Two special vectors: the **eigenvectors**. Two special scalars: the **eigenvalues**." The eigenvectors are the axes of the ellipse, the directions that $\widehat{\Sigma}$ only stretches. The eigenvalues are the stretch factors.

### 6.4 The math of PCA (p.63–74). Transcribed faithfully.
- **p.63** Sample covariance ($n\times n$), **dividing by T** (not T−1):
  $$\widehat{\Sigma}=\frac{1}{T}X'X\qquad\text{("Remember: } X \text{ has been centered.")}$$
  $\widehat{\Sigma}$ is symmetric but **might have rank less than n if, for instance, T < n.**
  Idea: find $\gamma\in\mathbb{R}^n$ such that the sample variance of $X\gamma$ is maximized.
- **p.64** Sample variance of the projection:
  $$\frac{1}{T}\gamma'X'X\gamma=\gamma'\widehat{\Sigma}\gamma.$$
  Without constraints, a large $\gamma$ gives infinite variance (not useful). So restrict to **unit vectors**: $\|\gamma\|=1 \iff \gamma'\gamma=1$.
- **p.65 First PC, the problem:**
  $$\max_{\gamma\in\mathbb{R}^n}\ \gamma'\widehat{\Sigma}\gamma\quad\text{s.t.}\quad\gamma'\gamma=1,\qquad L(\gamma,\lambda)=\gamma'\widehat{\Sigma}\gamma-\lambda(\gamma'\gamma-1),\ \lambda\in\mathbb{R}.$$
- **p.66 FOC:**
  $$\frac{\partial}{\partial\gamma_1}\Big[\gamma_1'\widehat{\Sigma}\gamma_1-\lambda_1(\gamma_1'\gamma_1-1)\Big]=\mathbf{0}\ \Rightarrow\ \widehat{\Sigma}\gamma_1-\lambda_1\gamma_1=\mathbf{0}\ \Rightarrow\ \widehat{\Sigma}\gamma_1=\lambda_1\gamma_1.$$
  (The slide drops the common factor 2 from the derivative. That is harmless.) This is an **eigenvector equation**: $\gamma_1$ is an eigenvector of $\widehat{\Sigma}$ and $\lambda_1$ is its eigenvalue. "Which pair should we choose?"
- **p.67** The objective at the solution is
  $$\gamma_1'\underbrace{\widehat{\Sigma}\gamma_1}_{=\lambda_1\gamma_1}=\gamma_1'\lambda_1\gamma_1=\lambda_1\gamma_1'\gamma_1=\lambda_1,$$
  so choose $\lambda_1$ as large as possible. **Solution: the eigenvector associated with the largest eigenvalue of $\widehat{\Sigma}$.** $Z_1:=X\gamma_1$ is the **first principal component (PC)** of X, and **its variance is $\lambda_1$**.
- **p.68 Second PC:** maximize $\gamma'\widehat{\Sigma}\gamma$ over unit $\gamma$ that is **(sample) uncorrelated with the first PC**.
- **p.69** Sample covariance of $X\gamma$ and $X\gamma_1$:
  $$\frac{1}{T}(X\gamma)'X\gamma_1=\gamma'\tfrac{1}{T}X'X\gamma_1=\gamma'\widehat{\Sigma}\gamma_1=\lambda_1\gamma'\gamma_1\quad(\Leftarrow\widehat{\Sigma}\gamma_1=\lambda_1\gamma_1).$$
  So "uncorrelated" ⇔ **orthogonal loadings**: $\gamma'\gamma_1=0$.
- **p.70**
  $$\max_{\gamma}\ \gamma'\widehat{\Sigma}\gamma\ \ \text{s.t.}\ \ \gamma'\gamma=1\ \text{and}\ \gamma'\gamma_1=0;\qquad L(\gamma,\lambda,\phi)=\gamma'\widehat{\Sigma}\gamma-\lambda(\gamma'\gamma-1)-\phi\gamma'\gamma_1.$$
- **p.71 FOC:**
  $$\frac{\partial}{\partial\gamma_2}\Big[\gamma_2'\widehat{\Sigma}\gamma_2-\lambda_2(\gamma_2'\gamma_2-1)-\phi\gamma_2'\gamma_1\Big]=\mathbf{0}\Rightarrow\widehat{\Sigma}\gamma_2-\lambda_2\gamma_2-\phi\gamma_1=\mathbf{0}.$$
  Left-multiply by $\gamma_1'$: $\gamma_1'\widehat{\Sigma}\gamma_2-\lambda_2\gamma_1'\gamma_2-\phi\gamma_1'\gamma_1=0 \Rightarrow 0-0-\phi\cdot1=0$. Hence **$\phi=0$** and $\widehat{\Sigma}\gamma_2-\lambda_2\gamma_2=\mathbf{0}$.
- **p.72** $\widehat{\Sigma}\gamma_2=\lambda_2\gamma_2$. Choose the eigenvector of the **second-largest eigenvalue**. $Z_2=X\gamma_2$ is the second PC, with sample variance $\gamma_2'\widehat{\Sigma}\gamma_2=\lambda_2\gamma_2'\gamma_2=\lambda_2$.
- **p.73 General j:** $\widehat{\Sigma}\gamma_j=\lambda_j\gamma_j$, with $\gamma_j$ orthogonal to all $\gamma_i$, $i=1,\dots,j-1$. Repeat until $j=n$, which gives up to n eigenvectors $\gamma_1,\dots,\gamma_n$ and eigenvalues $\lambda_1,\dots,\lambda_n$. **PCs: $Z_j:=X\gamma_j$, $j=1,\dots,n$.**
- **p.74 Keeping k PCs** ($1\le k\le n$):
  $$\Gamma_k:=(\gamma_1,\dots,\gamma_k)\ (n\times k),\qquad Z_{(k)}:=X\Gamma_k=(Z_1,\dots,Z_k)\ (T\times k).$$
  By construction the PCs are **orthogonal** with sample variances $\lambda_1,\dots,\lambda_k$, so
  $$\widehat{\mathrm{Cov}}(Z_{(k)})=\Lambda_k:=\mathrm{diag}(\lambda_1,\dots,\lambda_k).$$

### 6.5 PCA algorithm as taught (summary, all from p.45–74)
1. Put the data in a $T\times n$ matrix X (rows = observations/time, columns = variables).
2. **Demean each column** (p.45, p.63).
3. Compute $\widehat{\Sigma}=X'X/T$ (p.63).
4. Eigen-decompose $\widehat{\Sigma}$. Sort the eigenvalues $\lambda_1\ge\lambda_2\ge\dots$, with unit-norm, mutually orthogonal eigenvectors $\gamma_j$ (p.66–73).
5. Pick k (visual aid: the scree plot, p.43). Form $\Gamma_k$ and $Z_{(k)}=X\Gamma_k$ (p.74).
6. PC j has variance $\lambda_j$. The PCs are mutually uncorrelated (p.67, 72, 74).

### 6.6 Facts that follow but are NOT on the L1 slides. Flag them if you use them.
- Share of variance explained by PC j is $\lambda_j/\sum_{i}\lambda_i$ (not stated in L1).
- Link to the p.42 SVD: if $X=UDV^\top$ (full SVD), then $\widehat{\Sigma}=VD^2V^\top/T$, so $\gamma_j$ are the columns of V, $\lambda_j=d_j^2/T$, and the PC scores are $Z=XV=UD$. L1 shows the SVD only for the biplot and does not derive this link.
- L1 PCA uses the **covariance** matrix of **demeaned** data. It does **not** discuss scaling to unit variance (correlation-matrix PCA). PCA on covariance is scale-dependent. Standardization is a later-lecture or exam-notebook convention, not an L1 one.
- Software normalization: `sklearn.decomposition.PCA.explained_variance_` divides by $T-1$. `np.cov` defaults to $T-1$. The slide divides by $T$. Eigenvectors are identical. Eigenvalues differ by the factor $T/(T-1)$.
- **No code for PCA appears in L1.** There is no `sklearn.decomposition.PCA` and no `np.linalg.eig/eigh`.

## 7. Other high-dimensional data examples (L1 p.75–79). Examples only.
- **p.75–76 t-SNE** "visualizes **nonlinear** dimension reduction" (ref. van der Maaten & Hinton 2008, JMLR). The embedding of business-news articles is coloured by topic (Other, Economic growth, European sovereign debt, Bond yields, Macroeconomic data, Currencies/metals, Treasury bonds, Financial crisis, Mortgages, China). Source: Bybee, Kelly, Manela & Xiu (2024) "Business News and Business Cycles", JF 79(5). **Shown only. Not taught.**
- **p.77 Text data:** positive and negative sentiment word clouds (surge, climb, repurchase, surpass, undervalue… vs blame, plunge, downgrade, shortfall, disappointing, tumble…). Source: Ke, Kelly & Xiu (2021) "Predicting Returns with Text Data": 39,673 words, millions of articles.
- **p.78 Social media data:** a word cloud of political tweets (great, people, fake, news, border…).
- **p.79 Image data:** ImageNet hierarchy (mammal→placental→carnivore→canine→dog→working dog→husky; vehicle→craft→watercraft→sailing vessel→sailboat→trimaran). "The data that transformed AI research—and possibly the world."

## 8. The course's philosophy on AI tools, validation and benchmarks (L1 p.80–82). **Heavily exam-relevant.**
- **p.80 "What Has Changed, and What Has Not":** AI tools can now do the **mechanics** of data analysis end to end: load, notice a wrong header row, clean, plot, fit a model, describe the result, in about a minute, without being told the column names. "Now read that list again. **Not one item on it is the hard part.**"
  - Nothing there tells you **whether the answer is right**.
  - Nothing there tells you **whether the question was worth asking**.
  - Nothing there tells you **what would have to be true for the number to mean what you want it to mean**.
  - "**The same session that writes correct code will produce a confidently wrong answer in exactly the same tone. There is no tell.**"
- **p.81 "So What Is Scarce Now?"**
  1. **Framing.** What is the decision? What would count as an answer? "A forecast, a treatment effect and a trading signal are three different questions about the same data, and they need three different setups."
  2. **Validation.** "How would you know if you were fooling yourself? This is most of what the next eight lectures are about, and it is the part the machine is worst at, because **it requires knowing where your data came from**."
  3. **Judgment.** "**Is 0.72 good? Compared to what?** A tool will tell you it is good. It does not know your benchmark."
  - "That list is the syllabus… you should let the machine write the code the whole way. Lecture 9 closes the loop: the tool itself is assembled from exactly these parts."
- **p.82 "A Live Demonstration — and What to Watch For":** the professor gives an AI tool **the course's macroeconomic panel** and asks it to **predict industrial production growth**. Four things to watch:
  1. **Does it ask what the data are?** Or does it assume? It cannot see where your file came from, and it will not usually admit that.
  2. **How does it split the data?** "This is a **monthly time series**. **Watch whether the future ends up in the training set. It usually does.**"
  3. **What does it compare the answer to?** "**A number with no benchmark is not a result.**"
  4. **What happens when we push back?** Tell it the split was wrong and see whether it defends the answer or folds. Both are informative.
  - "You may use these tools on everything in this course. **Lecture 0 and `AI_Coding_Guide.pdf`** are about using them well. This slide is about why the rest of the course still exists."

---

## 9. Consolidated course conventions from L1
| Convention | Slide |
|---|---|
| Data matrix X is **T × n**: rows = observations/time, columns = variables. A PC is a T-vector (factor time series). Loadings are n-vectors. | p.42, p.45, p.73–74 |
| **Demean every column before PCA** ("X has been centered"). | p.45, p.63 |
| Sample covariance $\widehat\Sigma = X'X/T$ (**1/T**, not 1/(T−1)). | p.63 |
| Loadings are **unit-norm** ($\gamma'\gamma=1$) and mutually **orthogonal**. PCs are uncorrelated. PCs are ordered by eigenvalue. PC variance = eigenvalue. | p.64–74 |
| Correlation = covariance / (sd·sd). Use the correlation matrix and its **heat map** (`df.corr()`) as the first look at a wide dataset. | p.44 |
| Always pair ρ with a **scatter plot** (ρ captures only linear association). | p.44 |
| Supervised = regression (real output) or classification (class output). Unsupervised = patterns in unlabeled data (PCA is unsupervised). | p.25, p.46 |
| **Every result needs a benchmark** ("A number with no benchmark is not a result"; "Is 0.72 good? Compared to what?"). | p.81, p.82 |
| **Monthly time series: the future must not end up in the training set.** | p.82 |
| Framing first: forecast vs treatment effect vs trading signal need different setups. | p.81 |
| Validation requires knowing where the data came from. Ask what the data are. | p.81, p.82 |
| AI tools are allowed for everything. The student is responsible for validation and judgment. | p.81, p.82 |
| Class data lives in `/classes/41210_MiF_fall2026/` on Booth JupyterHub. | p.3 |
| Expect **low SNR** in return prediction (market efficiency). Finance samples are short (100s of months) and nonstationary. n ≈ p makes OLS infeasible. | p.33, p.34 |

## 10. Pitfalls and warnings emphasized in L1
| Warning | Slide |
|---|---|
| AI output is **confidently wrong in the same tone as when right. "There is no tell."** Mechanics are not the hard part. | p.80 |
| Tools assume rather than ask what the data are. They cannot see where the file came from. | p.82 |
| **Look-ahead in splitting:** with a monthly time series, "the future ends up in the training set. It usually does." | p.82 |
| **No benchmark means no result.** A tool "does not know your benchmark". | p.81, p.82 |
| Push back and see whether the answer is defended or folds. | p.82 |
| **ρ = 0 ≠ unrelated** (a U-shape gives ρ ≈ 0). Look at the scatter plot. | p.44 |
| Highly correlated columns carry little separate information (redundancy, and the motivation for PCA). | p.44, p.34 |
| **Finance has low SNR.** Market efficiency makes low predictability structural, so don't expect CS-like accuracy. | p.33 |
| **Small samples + nonstationarity** shrink effective sample size. **n ≈ p** makes OLS infeasible. | p.34 |
| Hype vs evidence: AI funds/ETFs (AIEQ lagging its comparison line; Sentient liquidated in under 2 years). Headlines flip within 3 weeks. | p.27, p.29–32 |
| Too-good-to-be-true, **simulated/levered** track records (smooth FS and "Simulated 3x FS"). | p.39 |
| Raw price series contain artefacts (the 4-1 split). Model returns and plot the data. | p.38 |
| PCA requires centered data. The variance objective needs the unit-norm constraint (else infinite variance). $\widehat\Sigma$ may be rank-deficient when T < n. | p.45, p.63–64 |

## 11. Datasets and worked examples in L1
- Email-validity toy table and linear "stats model" (p.17–18)
- MNIST handwritten 3s, 28×28 = 784 pixels, and a 784-16-16-10 NN diagram (p.19–23)
- Domo "Data Never Sleeps 5.0" (p.6). ML success timeline 1997–2022 (p.7–14)
- Press headlines 2017 (p.27). Morgan Stanley/FT ML-adoption survey 2016–2019 (p.28)
- AIEQ ETF and its Yahoo Finance performance, 2020–2024 (p.29–30). Sentient Investment Management (p.31–32)
- Gu–Kelly–Xiu (2019) ML long/short portfolios 1987–2016 vs SP500−Rf (p.36)
- Apple daily price and return 2016–2021 with the Aug-2020 4-for-1 split (p.38)
- S&P 100 / Lehman Bond Index / FS / Simulated 3× FS, Dec 1990–Dec 2007 (p.39)
- Walmart store locations (spatial) and expansion 1975–2005 (spatial-temporal), from Imai (p.40–41)
- High-frequency stock-return biplots, weeks of Mar 7–11 2005 and Mar 10–14 2008. Scree plot of about 90 eigenvalues and eigenvalue time series 2003–2013 (Aït-Sahalia & Xiu 2019) (p.42–43)
- Group-photo/camera projection illustration. 2-D point cloud with $Z=X\widehat\Sigma$ circle-to-ellipse mapping (p.47–62)
- t-SNE of business news (Bybee–Kelly–Manela–Xiu 2024) (p.75–76). Text sentiment word clouds (Ke–Kelly–Xiu 2021) (p.77). Social-media word cloud (p.78). ImageNet (p.79)
- Live demo: the course macro panel, predicting industrial production growth (p.82). The data are not shown.

## 12. Boundaries: mentioned only / not taught in L1
- **Neural networks** (p.23 diagram; NN3 in p.36 legend). Not taught.
- **Reinforcement learning** (p.25). Definition only.
- **t-SNE / nonlinear dimension reduction** (p.75–76). Shown only.
- **SVD**: used for the biplot formula $X\approx UDV^\top$ (p.42). Its link to PCA is not derived.
- **Scree plot** (p.43): shown. There is **no formal rule for choosing k** (no variance threshold, no CV, no information criterion in L1).
- GKX (2019) model menu **OLS-3+H, PLS, PCR, ENet+H, GLM+H, RF, GBRT+H, NN3** (p.36): legend labels only. **PLS, PCR, elastic net, GLM, GBRT and NN are not taught in L1.** (Trees/RF/boosting are L8. Ridge/lasso are L5. PCR, i.e. regression on PCs, is not taught in L1, which covers only the unsupervised PCA step.)
- The linear "stats model" for email validity (p.18). Not developed. Classification is L6.
- Recommender systems, NLP language modeling, text and image data (p.25, p.77–79). Examples only.
- Market efficiency / Fama (p.33). Domain-knowledge argument only, with no formal test.
- The "humans were the first machines" analogies (sorting, portfolios, Fama-French factors, priors as regularization; p.35). Conceptual only.
- **Not in L1 at all:** R²_OOS, cross-validation, train/validation/test design, expanding/rolling windows, standard errors, Sharpe ratio formula, MLE, penalties, trees/impurity, variable importance, standardization/scaling before PCA. These must be sourced from L2–L8.
- Lecture 0 and `AI_Coding_Guide.pdf` are referenced (p.82) as separate materials. Lecture 9 is referenced (p.81) but not in the local folder.

## 13. Mapping to the final exam
| Exam part | L1 slide(s) | How it applies |
|---|---|---|
| **1.1** majority-class baseline | p.81 ("Is 0.72 good? Compared to what?"), p.82 ("A number with no benchmark is not a result") | Justifies reporting "nobody purchases" first as the bar every model must clear. In the local CSV, 257 of 400 did not purchase, so the baseline accuracy is 257/400 = 64.25%. Recompute this in the notebook. |
| **1.2** describe the tree in plain English | p.37 (Design/Language: communicate to the audience), p.81 (Framing: what decision) | Supports writing the tree as a decision rule a non-statistician can use. The tree method itself is L8. |
| **1.3** RF/GB vs tree, given data size and shape | p.34 (ML earns its keep when there are many correlated variables, n ≈ p and unknown nonlinear form) | With n = 400 and only 3 features, the p.34 conditions for "ML is inevitable" hardly apply, so ensembles need not beat a small tree. The ensemble details are L8. |
| **1.4** impurity (in-sample) vs permutation (held-out) importance | p.81 (Validation: "how would you know if you were fooling yourself?"), p.44 (correlated columns share information) | Supports preferring the held-out measure for a decision maker, and warns that correlated features split credit. The importance mechanics are L8. |
| **1.5** shuffled CV on monthly stock data | **p.82** ("This is a monthly time series. Watch whether the future ends up in the training set. It usually does."), p.34 (nonstationarity), p.33 (low SNR) | Direct support. Shuffled folds put future months into training, which is look-ahead. Use time-ordered, forward-only evaluation (the procedure is from later lectures). The bias makes reported accuracy too **optimistic** ("fooling yourself", p.81). Low SNR means honest accuracy should sit near the base rate. |
| **2.1–2.3** self-training VaR pipeline | p.80 ("confidently wrong… no tell"), p.81 (validation "requires knowing where your data came from"), p.82 ("Does it ask what the data are?"), p.39 (too-good-to-be-true), p.63 (variance formula) | The pipeline never checks the provenance of its library, which is the p.81 validation failure. A VaR that keeps "de-risking" while markets are unchanged is a too-good-to-be-true signal, like p.39. **Caution: L1's $\widehat\Sigma$ divides by T, but Problem 2 explicitly mandates `var(ddof=1)`, i.e. 1/(n−1). Follow the exam.** L1 has nothing on martingales, log-variance drift or kurtosis, so source those elsewhere or derive them from first principles in the answer. |
| **2.2/2.3** plots | p.38 (time-series plot conventions), p.37 (statistics-first visualization) | Plot the log σ̂² path against the night and the histogram across desks. Annotate the key reference level (truth = 1) the way p.38 marks events. |
| **2.4(b)** real dj30 returns vs normal scenarios | p.33 (market returns dominated by news), p.44 (look at the data, not just one number), p.38 (daily return plot including a crisis spike) | Motivates looking at the real return distribution (March 2020) before trusting the normal fit. L1 gives no kurtosis or quantile formulas. |
| **3-features** know your data / correlation / duplicates | **p.44** (`df.corr()` heat map, the fastest way to see structure in a wide dataset; highly correlated columns carry little separate info), p.42 (biplot reveals **block structure** financial vs non-financial) | The notebook asks for the "correlation matrix ordered by class (block structure)". Use `df.corr()` plus a heat map. Screen the extended file x17..x164 for duplicate or near-duplicate columns (ρ ≈ 1; confirm exact duplicates with equality checks, since ρ = 1 also holds for affine copies). |
| **3-features** dimension reduction of many macro series | **p.45–74** (PCA algorithm), p.43 (scree plot for choosing k), p.34 (n ≈ p makes OLS infeasible; many correlated regressors), p.46 (PCA as "pre-treatment of the data for further ML algorithms", noise reduction) | PCA on the 148 extended (or 7 curated) country macro series, or on the global macro block, is an L1-grounded way to compress correlated predictors before a forecasting model. **Discipline (from p.82 plus the notebook rules): compute the means and eigenvectors on training-window data only, refit per window, and never use full-sample PCA.** Handle missing values first (PCA needs a complete matrix). L1 does not cover standardization before PCA. Because macro units differ, use the notebook's trailing standardization and state it as the exam's convention. |
| **3-features** characteristics sorts | p.35 ("nonlinearity: sorting returns by characteristics"; "dimension reduction: employing portfolios") | Grounds the notebook's "average next-month return of assets sorted into terciles" diagnostic as a classic nonlinear, portfolio-based view. |
| **3-evaluation** R²_OOS vs trailing mean and zero | p.81–82 (benchmark first; no future in training), p.33 (low SNR, so expect R²_OOS near 0 or negative), p.34 (100s of months, nonstationary) | Frames the expected result (small or negative R²_OOS is plausible and acceptable) and the trade-off between expanding windows (more data) and rolling windows (nonstationarity). The R²_OOS formula and the windows come from later lectures. |
| **3-evaluation** macro placebo | p.81 (Validation: "how would you know if you were fooling yourself?") | The circular-shift placebo is exactly a "fooling yourself" check. |
| **3-portfolio** forecast portfolios vs EW / risk parity / TSMOM | p.36 (GKX ML long-short portfolios vs SP500−Rf benchmark), p.29–32 (AI funds underwhelm in practice), p.39 (beware levered or simulated track records) | Supports a long/short sort on forecasts and comparison against simple benchmarks. Compare at comparable risk, since leverage alone inflates cumulative returns as in p.39. Report net of costs, because the AIEQ/Sentient experience says paper edges disappear. |
| **3-writeup** research paper | p.37 (visualization: statistics/design/language), p.80–81 (framing, validation, judgment), p.82 (ask what the data are; report the split and the benchmark) | The Data section should include EDA plots. The Methodology section must make the split and benchmark explicit. The Discussion should say what would have to be true for the numbers to mean what we claim (p.80). |
