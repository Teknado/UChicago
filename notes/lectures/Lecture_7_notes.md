# Lecture 7: Clustering (BUSN 41210, Dacheng Xiu, Sep 21 2026)

Source: `/home/user/UChicago/Lecture_7.pdf` (53 slides). I viewed every slide as a rendered image (PyMuPDF, 110 dpi;
p.51 and p.53 also at 300 dpi) and read the extracted text. Citations use the form "L7 p.X". X is both the PDF
page and the slide number.

**Scope in one sentence.** L7 teaches **unsupervised** learning, meaning models for **x** alone rather than E(y|x).
It covers:
- the mixture model and Gaussian mixture model (GMM, sklearn `GaussianMixture`, fitted by EM);
- the **K-means** objective and algorithm (sklearn `KMeans`), multiple random starts, and **standardizing** before
  clustering;
- choosing K by interpretation, with AIC/BIC as secondary tools (deviance = within-cluster sum of squares, df = K·p);
- K-means' implied model and its failure modes (rings, moons, unequal variances, anisotropic data);
- **spectral clustering** (similarity graph, graph Laplacian L = G − W, zero eigenvalues count connected pieces,
  K-means on the eigenvector rows, sklearn `SpectralClustering`);
- **agglomerative hierarchical clustering** and the dendrogram.

Worked examples: toy 2-D data, MNIST digits, U.S. House roll-call votes, WSJ news topics, and HFT trader clusters.

**What L7 does NOT cover (boundary):**
- **Linkage rules.** L7 never names single, complete, average or Ward linkage. It says only that "there are
  different ways to measure similarity" (p.48, p.52).
- **Hierarchical clustering code.** There is no `scipy.cluster.hierarchy` and no `AgglomerativeClustering`. The
  dendrogram on p.51 is a figure only.
- **Divisive clustering.** It is named (p.49) and then set aside: "Agglomerative strategies are simpler, we'll focus
  on them."
- **EM steps.** The EM algorithm behind GMM is named (p.13) but never derived. No E-step or M-step formulas appear.
- **GMM options.** `covariance_type`, GMM's `.bic()`/`.aic()` and `predict_proba` are not shown.
- **Other clustering metrics.** There is no silhouette score, elbow plot or gap statistic, and no external index
  such as ARI or NMI. The only external check is majority-label accuracy on MNIST (p.25, p.31).
- **Other algorithms.** DBSCAN, K-medoids, fuzzy clustering, t-SNE and UMAP do not appear.
- **Factor models.** They are named on p.3 as unsupervised, but L7 does not develop them. (PCA is L1.)
- **Scaling code.** There is no `StandardScaler` code in L7. Standardization appears only as a formula (p.23).
- **Time series and trees.** L7 contains nothing on time-series CV, R²_OOS, trees, ensembles, Sharpe ratios or VaR.
- **The trader clusters paper (p.7).** L7 shows only its figure, not its method.
- **The WSJ tree (p.53).** L7 shows only the figure and never explains how it was built.

---

## 0. Title and outline (L7 p.1–2)
- p.1: "BUSN 41210: Financial Analytics, Lecture 7: Clustering", Dacheng Xiu, Chicago Booth, September 21, 2026.
- p.2 Outline:
  1. Supervised versus unsupervised data analysis
  2. Model-based clustering: **mixture models**
  3. **K-means** algorithm
  4. Spectral clustering
  5. Hierarchical clustering

---

## 1. Supervised vs unsupervised; what clustering is (L7 p.3–8)

### p.3 Supervision
- "We've focused on models for E(y|x). Today is about models for **x**."
- "The goal in everything we do has been *Dimension Reduction*." DR means moving "from high dimensional x to low-D
  summaries."
- DR can be supervised or unsupervised.
  - **Supervised: regression and classification.** "HD x is projected through β into 1D ŷ." "Outside info (y)
    supervises how you simplify x."
  - **Unsupervised: mixture and factor models.** "x is modeled as built from a small number of components."
    "You're finding the simplest representation of x alone."
- "We always want the same things: **low deviance, without overfit**." Deviance is the fit criterion throughout the
  course; for K-means it becomes the within-cluster sum of squares (p.18, p.30).

### p.4 What clustering is, and why
- Definition: "dividing up data into groups (clusters), so that points inside each group are more 'similar' to each
  other than to points outside the group."
- There are two uses:
  - **Summary (DR):** "deriving a reduced representation of data."
  - **Discovery:** "looking for new insights into the data structure. Clustering can also help with predictions."
- **Warning, in emphasized text: "clustering should not be confused with classification!"**
  - "In classification, we have data for which the groups are **known** and we try to learn what differentiates
    them to assign future labels."
  - "In clustering, we have data for which the group labels are **unknown** and try to learn the groups themselves
    as well as what differentiates them."

### p.5 Clustering as unsupervised dimension reduction
- "Group observations into similar 'clusters', and understand the rules behind this clustering."
- Examples:
  - Demographic clusters: soccer moms, NASCAR dads.
  - Consumption clusters: jazz listeners, classic rock fans. **Collaborative filtering** is the idea of grouping
    individuals into clusters and modelling the average behaviour of each.
  - Industry clusters: competitor groups, supply chains.
- "Clustering is largely an **exploratory** technique." It comes in three families:
  - **model-based** methods (mixture models);
  - **nonparametric** methods (spectral clustering);
  - **"heuristic"** methods (hierarchical clustering). "Sometimes, it is useful to have clusters organized in a
    hierarchy."

### p.6 Applications
- **Marketing:** find distinct groups in the customer base and target marketing programs at them.
- **Land use:** find areas of similar land use in earth-observation data.
- **Insurance:** find groups of motor policy holders with high average claim cost.
- **Urban planning:** group houses by type, value and location.
- **Earthquake studies:** epicenters should cluster along continental faults.
- **Finance:** "Identify groups of stocks with similar characteristics to create portfolios that try to profit
  from differences within/across groups." This is the finance motivation.

### p.7 Clustering traders (figure only)
- Source: Aït-Sahalia and Brunetti, *Journal of Econometrics* 217 (2020) 20–45, Fig. 4. "Clustering analysis:
  trading frequency and inventory held by different types of traders, August 2010."
- Axes: x is carried-over **inventory** (−1 to 1); y is **trading frequency** (×10⁻³, 0 to 5).
- The four clusters, drawn as ellipses:
  - **High Frequency Traders** (red): tall and thin, inventory ≈ 0, frequency roughly 1–5×10⁻³.
  - **Fundamental Sellers**: inventory ≈ −0.6, low frequency.
  - **Residual Traders**: inventory ≈ 0, low frequency.
  - **Fundamental Buyers**: inventory ≈ +0.6, low frequency.
- Caption: LFTs are in the blue shaded areas; individual HFT accounts are not shown, for anonymity.
- The figure shows clusters of market participants defined on two continuous features. The method is not described.

### p.8 What is good clustering?
- A good method produces clusters with **high intra-class similarity** and **low inter-class similarity**.
- "The quality of a clustering result depends on both the **similarity measure** used by the method and on its
  implementation."
- Quality is also measured by the method's "ability to discover potential hidden patterns."

---

## 2. Mixture models and the Gaussian mixture model (L7 p.9–14)

### p.9 The K-means mixture model
- "The fundamental model of clustering is a **mixture**: observations are random draws from K populations, each with
  different average characteristics."
- Conditional mean:
  $$\mathbb{E}[\mathbf{x}_i \mid k_i] = \boldsymbol\mu_{k_i},\qquad k_i\in\{1,\dots,K\}$$
  For example, if k_i = 1, then x_i comes from cluster 1 and E[x_{i1}] = μ_{11}, …, E[x_{ip}] = μ_{1p}.
- "Each mean μ_j has an associated probability or 'weight' in the mixture."
- For a new x whose k is unknown:
  $$\mathbb{E}[\mathbf{x}] = p(k=1)\boldsymbol\mu_1+\dots+p(k=K)\boldsymbol\mu_K$$
- DR: "Given μ_k's, you discuss data in terms of K different types, rather than trying to imagine all possible values
  for each x."

### p.10 Mixture without knowing membership k (figure)
- A 1-D density on x from 5 to 37. It has a small mode near 10, a large mode near 20 with a shoulder near 23, and a
  small bump near 33.
- "The *marginal* density has multiple modes; one for each μ_k."

### p.11 Breaking into K components (figure)
- The same density split into four coloured components: blue ≈ 10, green ≈ 20, red ≈ 23, cyan ≈ 33. The dashed
  grey line is the marginal density.
- "Here, we have K = 4 different cluster centers. **Should it be 5?**" This foreshadows the difficulty of choosing K
  (p.29–31).

### p.12 Gaussian mixture model
- A figure shows three blue Gaussian components and their orange sum (a trimodal density).
- "Assuming each data point is generated with a certain probability π_j from K Gaussian distributions with
  N(μ_j, Σ_j), j = 1, 2, …, K." Written out, the density is:
  $$p(\mathbf{x})=\sum_{j=1}^K \pi_j\,\mathcal N(\mathbf{x}\mid\boldsymbol\mu_j,\Sigma_j),\qquad \sum_j\pi_j=1$$
  (The slide states this in words; the equation is my rendering of it.)
- **Warning:** "Likelihood is not unimodal and may stuck at local minimums." So the answer depends on where the fit
  starts, just as with K-means.

### p.13 GMM code (**house style**)
- "The GaussianMixture object implements the **expectation-maximization (EM)** algorithm for fitting
  mixture-of-Gaussian models."
```python
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
gmm_clusters = gmm.predict(X)
```
- Fitted attributes, as quoted on the slide:
  - `weights_`: "The weights of each mixture components" (the π_j).
  - `means_`: "The mean of each mixture component" (the μ_j).
  - `covariances_`: "The covariance of each mixture component" (the Σ_j).
- No `random_state`, `covariance_type` or `n_init` is set. All other arguments are sklearn defaults.

### p.14 GMM example (figure)
- Left panel, "True cluster": three 2-D clusters with **very different spreads**.
  - Red: tight, around (−9, −5.5).
  - Green: large and diffuse, around (−4, 0).
  - Blue: very tight, around (2, 0.5).
- Right panel, "GaussianMixture cluster": GMM recovers all three groups. The colours are permuted, which shows that
  cluster labels are arbitrary.
- This is **the same data as the K-means failure on p.35 ("Unequal Variances")**. GMM succeeds there because each
  component has its own Σ_j.

---

## 3. K-means: objective, algorithm, sklearn, scaling (L7 p.15–23)

### p.15 The K-means method
- It is "one of the oldest but still very popular clustering method due to its simplicity."
- "It applies to situations where all variables are of the **continuous** type (or are transformed to)."
- The dissimilarity is the **squared Euclidean distance**:
  $$d_{i,j}=\|\mathbf{x}_i-\mathbf{x}_j\|^2$$

### p.16 The K-means criterion (objective function)
- "Let K ≥ 1 be the number of clusters (**chosen a priori**)." For an assignment **k**, the within-cluster variation
  is:
  $$W(\mathbf{k})=\sum_{k=1}^K\sum_{k_i=k}\sum_{k_j=k}\|\mathbf{x}_i-\mathbf{x}_j\|^2$$
- The slide says this is proportional to:
  $$\sum_{k=1}^K N_k\sum_{k_i=k}\|\mathbf{x}_i-\boldsymbol\mu_k\|^2$$
  Here N_k is the number of points in cluster k and μ_k is its mean, called the **centroid**.
- *My note:* the exact identity is Σ_{i,j∈C_k}‖x_i−x_j‖² = 2N_k Σ_{i∈C_k}‖x_i−μ_k‖², as in ESL (14.31).
  - The algorithm (p.18) and sklearn's `inertia_` (p.22) both use the **unweighted** sum Σ_k Σ_{i∈C_k}‖x_i−μ_k‖².
  - The N_k weighting on p.16 comes from the pairwise form. The slides do not dwell on this difference.

### p.17 The chicken-and-egg problem
- For a given K, the goal is small within-cluster variability. We know neither the memberships k_i nor the
  centroids μ_k.
- But:
  1. If we knew the k_i, we could easily estimate the μ_k.
  2. If we knew the μ_k, we could easily estimate the k_i.
- "*Solution*: iterate between 1. and 2."

### p.18 The K-means algorithm
- "The K-means algorithm looks for an assignment k that minimizes W(k)."
- **Chicken:** if you know the membership k_i of each x_i, estimate each centroid as
  $$\hat{\boldsymbol\mu}_k=\frac{1}{n_k}\sum_{i:k_i=k}\mathbf{x}_i$$
  where {i : k_i = k} are the n_k observations in group k.
- **Egg:** if you know the means μ_k, find k = k_1…k_n to minimize the sum of squares
  $$\sum_{k=1}\sum_{i:k_i=k}(\mathbf{x}_i-\hat{\boldsymbol\mu}_k)^2$$
- "**Mixture deviance**: total sums of squares **within** each cluster."

### p.19 "Give K-means the x_i's, and it gives you back the k_i's"
- The figure shows three clusters in (X1, X2) with red diamond centroids.
- In words:
  - "Label each point based on the closest centroid (mean)."
  - "Replace each centroid by the average of the points in the cluster."
- "The algorithm starts at random μ_k, and changes k_i's until the sum of squares stops improving."
- **"Solution depends on start location. Try multiple, take the best answer."**
- **"Choosing K: For most applications, everything is descriptive. So try a few and use clusters that make sense to
  you."**

### p.20 K-means example (figure)
- Setup: X_i = (X_{i1}, X_{i2}), **n = 300, K = 3**.
- Six panels:
  - initial centers (random);
  - Iteration 1: WCV = 70.1;
  - Iteration 2: WCV = 65.72;
  - Iteration 3: WCV = 55.33;
  - Iteration 9: WCV = 24.44;
  - Iteration 10: WCV = 24.44 (converged).
- WCV means within-cluster variation. It falls monotonically, and the algorithm stops when it stops improving.

### p.21 K-means example with multiple runs (figure)
- Setup: X_i = (X_{i1}, X_{i2})′, **n = 250, K = 4**. "The points are not as well-separated."
- Three runs from different random starts: **WCV = 22.18, 16.13 and 18.23**.
- The slide explains that the runs use "different initial centers (chosen randomly over the range of the X_i's). We
  choose the second collection of centers because it yields the **smallest within-cluster variation (mixture
  deviance)**."
- **Course rule:** among random restarts, keep the run with the lowest within-cluster SS.

### p.22 KMeans code (**house style**)
- "Clusters X (**numeric!**) into n_clusters groups."
```python
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3,init='random',n_init=1,max_iter=10)
kmeans.fit(X)
```
- Output each cluster's center coordinates:
```python
np.around(kmeans.cluster_centers_,2)
# array([[ 0.  ,  0.03],
#        [-0.  ,  0.99],
#        [ 1.01,  0.97]])
```
- Cluster label counts:
```python
from collections import Counter
Counter(kmeans.labels_)
# Counter(0: 101, 1: 105, 2: 94)
```
  101 + 105 + 94 = 300, so this is the p.20 example.
- Within-cluster sum of squares:
```python
round(kmeans.inertia_,2)
# 24.44
```
  This matches the converged WCV on p.20.
- House-style kwargs: `init='random'`, **`n_init=1`**, `max_iter=10`.
  - These settings are for teaching: one random start, so you can watch the iterations on p.20.
  - Per p.19 and p.21, real use should try multiple starts. In sklearn that means `n_init` > 1, which keeps the
    lowest-inertia run automatically.
- Attributes used: `cluster_centers_`, `labels_`, `inertia_`.
- No `random_state` is shown.

### p.23 Scaling for K-means (**key convention**)
- "The algorithm minimizes total [squared] distance from center, *summed across all dimensions of x*."
- "**Scale matters**: if you replace x_j with 2x_j, that dimension counts twice as much in determining distance from
  center (and will have more influence on cluster membership)."
- "Standard solution is to **standardize**: cluster on scaled
  $$\tilde x_{ij}=\frac{x_{ij}-\bar x_j}{\mathrm{sd}(x_j)}$$"
- This is the only standardization formula in L7. It is column-wise (each feature j), using the mean and sd across
  observations.

---

## 4. MNIST case study (L7 p.24–28)

### p.24 The MNIST database
- The database is from Yann LeCun's website: a training set of **60,000** examples and a test set of **10,000**.
  (The slide title misspells it "MINST".)
- The digits are size-normalized and centered in a fixed-size image.
- Training data is a **60,000 × 784** matrix; test data is **10,000 × 784** (28×28 pixels).
- The figure shows 15 sample digits.

### p.25 10-means clustering on MNIST
- "The most natural choice of K is 10. **Why?**" Because there are 10 digits.
- The slide displays the 10 cluster centers as images. Their majority labels are **2, 6, 1, 8, 3, 4, 0, 1, 7, 9**.
  - Two clusters have majority label 1, and no cluster has majority label 5.
  - The "4" centroid looks like a 4/9 blend.
- **Cluster-then-label classifier:** "If using the **majority label in the clusters as the predicted label**, the
  model has the accuracy of: **57.84 % on training data and 59.45 % on testing data**."
- This is the only place L7 evaluates clusters against known labels, and it uses the given train/test split. The
  test accuracy comes from assigning test points to the clusters fitted on training data.

### p.26 The cluster whose majority label is 6
- Ten random images. Their true labels are 6, 6, 6, 6, 6, 6, 6, 6, **4**, 6: nine sixes and one 4.

### p.27 The cluster whose majority label is 1
- Ten random images. Their true labels are 3, 1, 1, 7, 3, 1, 1, 1, 8, 9: only five of ten are 1s.
- This cluster is impure. It groups thin, slanted strokes of several digits.

### p.28 xkcd cartoon
- "Our analysis shows that there are three kinds of people in the world: those who use K-means clustering with K=3,
  and two other types whose qualitative interpretation is unclear."
- This is a warning about arbitrary choices of K and about over-interpreting clusters.

---

## 5. Choosing K (L7 p.29–31)

### p.29 Choosing K
- "**1st order advice**: most often clustering is an exploration exercise, so choose the K that makes the most sense
  to you."
- "But, we *can* apply data-based model building here:
  1. Enumerate models for k_1 < k_2 < … < k_K.
  2. Use a selection tool to choose the best model for new x."
- "Step one is easy. Step two is a tougher."
- "For example, for CV you'd want to have high OOS p_{k_i}(x_i). But you don't know k_i! This is a *latent*
  variable. **There's no ground truth like y to compare against.**"
- So cross-validation is not straightforward for clustering.

### p.30 AIC and BIC for K-means
- "We *can* use IC to select K."
  - "Deviance is (as always) D = −2 log LHD, which for K-means is the **total sum of squares** (slide 10)." This is
    the within-cluster SS.
  - "**df is the number of μ_kj: K × p**" (where p is the dimension of x).
- "Then our usual AIC and BIC formulas apply."
  - *Cross-reference:* L4 p.29–30 gives the general form deviance + k·df: AIC = D + 2·df and BIC = D + log(n)·df.
  - For K-means this gives AIC = WCSS + 2Kp and BIC = WCSS + Kp·log(n).
  - The slide does not write these out. The "usual formulas" refer back to L4.
- "**Beware**: the assumptions behind both AIC and BIC calculations are only roughly true for K-means."
- "These tools are lower quality here than in regression."
- "**You're often better off just using descriptive intuition.**"

### p.31 BIC and AIC for MNIST K-means (figures)
- The BIC curve spans K = 10 to 50, with values of about 2.45×10⁶ falling to 2.25×10⁶. Its minimum is at **K = 30**.
- The AIC curve spans K ≈ 10 to 190, falling from about 2.35×10⁶ to 1.83×10⁶. Its minimum is at **K = 130**.
- "BIC likes K = 30, AIC likes 130. **Both are way more complicated than is useful.**"
- "With K = 30, the model has the accuracy of: **75.085% on training data, and 76.29% on testing data**." That is up
  from 57.84% / 59.45% with K = 10.
- Takeaway: more clusters give higher majority-label accuracy, but interpretability falls. AIC picks a larger K than
  BIC, consistent with L4 ("AIC prefers more complicated models").

---

## 6. The model behind K-means, and when it fails (L7 p.32–37)

### p.32 Revisiting the K-means model
- "In minimizing sums-of-squares, K-means targets the model
  $$p_k(\mathbf{x})=\prod_j \mathrm N(x_j\mid\mu_{kj},\sigma^2)$$"
- It therefore assumes "**independence across dimensions** (no multicollinearity) and **uniform variance** (same σ
  for all j, **which is why scale matters**)."
- "Despite being a silly model, this tends to do a decent job of clustering when x consists of *continuous*
  variables."
- "**It does a worse job for x made up of dummies or counts.**"
- K-means is thus a special case of the GMM on p.12: equal weights in effect, Σ_j = σ²I for every cluster, and hard
  assignment.

### p.33 K-means fails on non-Gaussian shapes: two rings (figure)
- True clusters are an outer ring (radius ≈ 1) and an inner ring (radius ≈ 0.5).
- The K-means result puts two centroids (black X) at about (−0.3, 0.4) and (0.3, −0.4). It splits the plane in half
  diagonally, and each cluster takes half of each ring.

### p.34 K-means fails on non-Gaussian shapes: two moons (figure)
- True clusters are two interleaved half-moons. K-means (centroids ≈ (−0.2, 0.58) and (1.2, −0.08)) cuts them with
  a straight boundary.

### p.35 K-means fails on unequal variances (figure)
- This is the p.14 data: a tight cluster, a diffuse cluster and a very tight cluster.
- K-means places centroids near each group but assigns the diffuse cluster's outer points to the tight neighbours.
  The boundaries are linear bisectors, which ignore spread.
- Contrast: GMM gets this right (p.14).

### p.36 K-means fails on anisotropic data (figure)
- True clusters are three parallel, elongated diagonal strips.
- K-means cuts across the strips: two centroids split the two left strips horizontally rather than along their axes.

### p.37 Why K-means fails on these shapes
- "K-means assigns each point to the nearest **center**: it uses straight-line distance to a single point."
- On the two rings: "Two points on the *same* outer ring are on average **1.28** apart; a point on the outer ring and
  a point on the inner ring are on average **1.05** apart. Measured this way, the rings are not clusters at all."
- "But every point on the outer ring is close to its **neighbors**, and through them it is linked to every other
  point on that ring. **Similarity is local**: a cluster is a set of points connected by chains of nearby points."
- "So build the graph of who is near whom, and cluster the graph instead of the coordinates."

---

## 7. Spectral clustering (L7 p.38–47)

### p.38 The similarity graph
- The graphic defines a similarity graph G(V, E, W):
  - V: the vertices (data points);
  - E: an edge wherever similarity > 0;
  - W: the edge weights (similarities).
- The panels show the data (three coloured groups), the similarity matrix as a block pattern, and the graph.
- "Each data point is a vertex. Connect i and j if they are similar, with weight **W_{i,j} ≥ 0; W is symmetric**."
- **K-nearest-neighbor graph:** W_{i,j} = 1 if i is among the K nearest neighbours of j *or* j is among those of i;
  otherwise W_{i,j} = 0.
- **Fully connected graph:** $W_{i,j}=\exp(-d^2(i,j)/c)$, with d(i,j) a distance. This is a Gaussian kernel with
  bandwidth c.
- "Two rings, 10-nearest-neighbor graph: **no edge between the rings**. In the graph, the rings are two separate
  pieces."

### p.39 The graph Laplacian
- "Let G be diagonal with $G_{i,i}=\sum_j W_{i,j}$, the total weight at vertex i, and define
  $$L=G-W.$$"
- Worked example: a six-point graph made of two triangles, with no edge between them.
  $$W=\begin{pmatrix}0&1&1&0&0&0\\1&0&1&0&0&0\\1&1&0&0&0&0\\0&0&0&0&1&1\\0&0&0&1&0&1\\0&0&0&1&1&0\end{pmatrix},\quad
  L=\begin{pmatrix}2&-1&-1&0&0&0\\-1&2&-1&0&0&0\\-1&-1&2&0&0&0\\0&0&0&2&-1&-1\\0&0&0&-1&2&-1\\0&0&0&-1&-1&2\end{pmatrix}$$
- "Two facts, each one line":
  - "Every row of L sums to zero, so **L𝟏 = 0**: the constant vector has eigenvalue 0."
  - $$v^\top L v=\tfrac12\sum_{i,j}W_{i,j}(v_i-v_j)^2\ \ge 0$$ "for every v, so every eigenvalue of L is ≥ 0."

### p.40 Zero eigenvalues count the pieces
- "v⊤Lv = 0 exactly when v_i = v_j for every edge W_{i,j} > 0, that is, when v is **constant on each connected
  piece** of the graph."
- "So if the graph has K separate pieces, L has exactly K eigenvectors with eigenvalue 0: the indicators of the
  pieces (or any rotation of them)."
- The figures show the graph (triangles {1, 2, 3} and {4, 5, 6}), the eigenvalues plotted by index (two zeros,
  then four at 3), and the rows of V in the (v1, v2) plane. The rows collapse to two locations: points 1–3 at about
  (0.577, 0) and points 4–6 at about (0, 0.577).
- "Six-point graph: eigenvalues **0, 0, 3, 3, 3, 3**; the two eigenvectors with eigenvalue 0 are
  (1,1,1,0,0,0)/√3 and (0,0,0,1,1,1)/√3. As columns of V: **row i of V is the same for every point in a piece**.
  Six points become two locations."

### p.41 The spectral clustering algorithm (**the algorithm as taught**)
1. "Build the similarity matrix W (K-nearest neighbors, or a Gaussian kernel)."
2. "Form L = G − W."
3. "Compute the K eigenvectors of L with the *smallest* eigenvalues, V_{n×K} = [v_1, …, v_K]."
4. "Represent point i by row i of V: the K numbers (v_{1,i}, …, v_{K,i})."
5. "Run K-means on these n new points."

- "**Why K-means comes back.** If the graph has exactly K pieces, the rows of V take K distinct values and K-means
  has nothing to do. Real clusters are only *nearly* disconnected: the K smallest eigenvalues are small rather than
  zero, the rows of V are nearly constant on each cluster, and K-means finishes the job."
- **Code (house style):** "In Python: `SpectralClustering(n_clusters=K, affinity='nearest_neighbors')`. It uses a
  **normalized** version of L (rows divided by the G_{i,i}); the idea is the same."
  - The import is not shown. It is `from sklearn.cluster import SpectralClustering`.
  - "Rows divided by G_{i,i}" means G⁻¹L = I − G⁻¹W, the random-walk normalized Laplacian. That label is mine; the
    slide gives no name.
- *Contrast with L1 PCA (my note):* PCA keeps the eigenvectors of the covariance matrix with the **largest**
  eigenvalues. Spectral clustering keeps the eigenvectors of the Laplacian with the **smallest** eigenvalues.

### p.42 Two rings, worked
- The panels show the 10-NN graph (blue outer ring, red inner ring), the eigenvalues of L for indices 1–12, and the
  rows of V.
  - The eigenvalues rise from 0 to about 0.31.
  - In the rows-of-V plot, the 300 outer-ring points sit at about (0, −0.058) and the 300 inner-ring points at
    about (0.058, 0).
- "600 points, 10-nearest-neighbor graph: no edge crosses between the rings, so the graph has two pieces."
- "Eigenvalues of L: **0, 0, 0.031, 0.036, …** — two zeros, then a gap. **The number of eigenvalues near zero tells
  you K.**" This is the eigengap heuristic for choosing K.
- "The rows of V sit at two locations. K-means on them recovers the rings **exactly**; K-means on the original
  coordinates gets **51%** right."

### p.43 Spectral clustering vs K-means: rings and moons (figures)
- Rows show rings (top) and moons (bottom). Columns show the true cluster, the K-means cluster (with X centroids)
  and the spectral cluster.
- Spectral clustering recovers both rings and both moons. The colours are swapped relative to the truth, which
  doesn't matter.

### p.44 Spectral clustering vs K-means: anisotropic data and unequal variances (figures)
- Top row, anisotropic strips: K-means cuts across the strips, while spectral clustering separates the three strips
  correctly.
- Bottom row, unequal variances: K-means mis-assigns the diffuse cluster's edges. Spectral clustering gives the
  diffuse cluster, the tight lower-left cluster and the tight right cluster, close to the truth.
- Visually, spectral clustering fixes all four K-means failure cases shown on p.33–36.

### p.45 Example: clustering legislators on votes
- The data are U.S. House of Representatives roll-call votes. Each line is one member, with results on **669** roll
  calls: **1 = Yea, −1 = Nay, 0 = no vote**.
- There are **434** representatives.
- "Goal: We use voting behavior to reveal legislators' political view (liberal or conservative)."

### p.46 Clustering results
- The left figure shows eigenvalues for indices 1–10: 0, then about 0.66, then a plateau at about 0.98–1.0.
- The right figure shows the second eigenvector sorted across the 434 representatives, as a bar chart coloured by
  party.
  - About the first 200 bars are negative, from about −0.004 up to 0: the Democrats (blue).
  - The rest are positive, up to about 0.003: the Republicans (red).
  - The legend also lists Independent (green).
- "**Fully connected graph**, $W_{i,j}=\exp(-d_{i,j}/669)$ with d_{i,j} the number of roll calls on which i and j
  differ."
  - Here the kernel uses d (not d²) and the bandwidth c = 669, the number of roll calls. So d/669 is the fraction
    of disagreements.
- "The graph is connected, so exactly one eigenvalue is 0 — the constant vector, which says nothing. The eigenvalues
  are **0, 0.66, 0.98, 0.98, …**: one small one after the zero, then a plateau. **Two groups.**"
  - Eigenvalues near 1 are consistent with the normalized Laplacian.
- "The **second eigenvector** gives one coordinate per member. **Its sign matches the party of all 433 Democrats and
  Republicans.**"

### p.47 Clustering results: the two ends of the second eigenvector
- A table (`Name`, `second_eig`, `Group`; "434 rows × 3 columns") lists the most extreme members.
  - Most negative, all Democrats:
    - Schakowsky −0.065527
    - Woolsey −0.064166
    - Lee −0.064044
    - Baldwin −0.063970
    - Miller, George −0.063947
  - Most positive, all Republicans:
    - Blackburn 0.049626
    - Foxx 0.049632
    - Akin 0.049699
    - McHenry 0.050037
    - Neugebauer 0.050160
- The scale here differs from the plot on p.46. The slide does not explain why; it may use a different
  normalization.
- "The coordinate **orders** members from one end of the House to the other, not just into two groups." So the
  eigenvector is a continuous 1-D summary, which is dimension reduction.

---

## 8. Hierarchical clustering (L7 p.48–53)

### p.48 From K-means to hierarchical clustering
- Two properties of K-means:
  - "It fits exactly K clusters (as specified)."
  - "Final clustering assignment depends on the chosen initial cluster centers."
- "**Hierarchical clustering** is an alternative that **does not rely on any underlying model**."
  - "Hierarchical clustering produces a sequence of **nested** cluster memberships."
  - "No need to choose initial starting positions and the number of clusters."
  - "Data points that are similar will end up in the same cluster."
- "There are different ways to measure similarity." No specific ways are listed.
- "At one end, all points are in their own cluster, at the other end, all points are in one cluster."

### p.49 Agglomerative vs divisive
- **Agglomerative (bottom-up):** start with every point in its own group. "Until there is only one cluster,
  repeatedly: **merge the two groups that have the smallest dissimilarity**."
- **Divisive (top-down):** start with all points in one cluster. "Until all points are in their own cluster,
  repeatedly: split the group into two resulting in the biggest dissimilarity."
- "Agglomerative strategies are simpler, **we'll focus on them**."

### p.50 A simple example
- Seven 2-D points, read from the plot:
  - 0 ≈ (0.00, 0.52)
  - 1 ≈ (0.55, 0.49)
  - 2 ≈ (0.77, 0.16)
  - 3 ≈ (0.77, 0.02)
  - 4 ≈ (0.14, 0.12)
  - 5 ≈ (0.31, 0.68)
  - 6 ≈ (0.47, 0.82)
- The agglomerative sequence:
  - Step 1: {0}, {1}, {2}, {3}, {4}, {5}, {6}
  - Step 2: {0}, {1}, {2, 3}, {4}, {5}, {6}
  - Step 3: {0}, {1}, {2, 3}, {4}, {5, 6}
  - Step 4: {0}, {1, 5, 6}, {2, 3}, {4}
  - Step 5: {0, 4}, {1, 5, 6}, {2, 3}
  - Step 6: {0, 4, 1, 5, 6}, {2, 3}
  - Step 7: {0, 1, 2, 3, 4, 5, 6}

### p.51 The dendrogram (figure)
- "We can also represent the sequence of clustering assignments as a **dendrogram**." Beside it is the p.50 scatter.
- The leaf order is 2, 3, 1, 5, 6, 0, 4. Merge heights, read at 300 dpi:
  - {2, 3} ≈ 0.14
  - {5, 6} ≈ 0.22
  - {1, 5, 6} ≈ 0.35
  - {0, 4} ≈ 0.43
  - {1, 5, 6, 0, 4} ≈ 0.78
  - root ≈ 1.08
- "Note that **cutting the dendrogram horizontally** partitions the data points into clusters." For example, a cut
  at 0.6 gives {2, 3}, {1, 5, 6} and {0, 4}.
- *My check, not stated on the slide:* with the coordinates above, these heights match **Ward linkage** in scipy's
  convention.
  - Singletons merge at their Euclidean distance: d(2,3) = 0.14, d(5,6) = 0.21, d(0,4) = 0.42.
  - Ward gives {1} + {5, 6} = 0.35, {0, 4} + {1, 5, 6} = 0.78, and the root = 1.09.
  - Complete linkage would put the root at about 0.92.
  - So the figure was probably made with `scipy.cluster.hierarchy.linkage(X, 'ward')`. The lecture never names a
    linkage, so this is an inference.

### p.52 What's a dendrogram?
- It is a "convenient graphic to display a hierarchical sequence of clustering assignments." Simply a tree where:
  - each node represents a group;
  - each leaf node is a singleton (a group containing a single data point);
  - the root node is the group containing the whole data set;
  - each internal node has two children, the groups that were merged to form it.
- "Remember: **the choice of similarity measure determines how we merge groups of points**."

### p.53 Hierarchical clustering of WSJ news
- "We have data from **180 topics**, each being a vector of length **18,433**." These are topic-word vectors from
  WSJ articles.
- The figure is a horizontal tree. Branches include:
  - Banks: bank loans, credit ratings, nonperforming loans, savings & loans, financial crisis, …
  - Asset Managers & I-Banks: accounting, NASD, acquired investment banks, private equity/hedge funds, mutual funds.
  - Buyouts & Bankruptcy: real estate, Drexel, control stakes, M&A, corporate governance, bankruptcy, SEC,
    takeovers, convertible/preferred.
  - Financial Markets: exchanges/composites, options/VIX, commodities, currencies/metals, international exchanges,
    trading activity, small caps, Treasury bonds, bond yields, bear/bull market, share payouts, IPOs, short sales.
  - Corporate Earnings: earnings losses, earnings, profits, earnings forecasts, financial reports, revised estimate,
    small changes.
  - Economic Growth: European sovereign debt, Federal Reserve, economic growth, macroeconomic data, recession,
    record high, optimism, product prices, …
  - Financial Intermediaries is a higher-level node joining these groups, and "Economics" is near the root.
- "See full graph on this website."
- Source: **Bybee, Kelly, Manela, and Xiu, "Business News and Business Cycles" (2024, *Journal of Finance*)**.
- Takeaway: hierarchical clustering organizes a large set of high-dimensional series (topics) into interpretable
  nested themes.

---

## 9. Consolidated formula sheet (all L7)

| Object | Formula | Slide |
|---|---|---|
| Mixture conditional mean | E[x_i \| k_i] = μ_{k_i}, k_i ∈ {1..K} | p.9 |
| Mixture marginal mean | E[x] = Σ_k p(k) μ_k | p.9 |
| GMM | x ~ Σ_j π_j N(μ_j, Σ_j) | p.12 |
| K-means dissimilarity | d_ij = ‖x_i − x_j‖² | p.15 |
| Within-cluster variation | W(k) = Σ_k Σ_{k_i=k} Σ_{k_j=k} ‖x_i − x_j‖² ∝ Σ_k N_k Σ_{k_i=k} ‖x_i − μ_k‖² | p.16 |
| Centroid update | μ̂_k = (1/n_k) Σ_{i:k_i=k} x_i | p.18 |
| Assignment step | minimize Σ_k Σ_{i:k_i=k} (x_i − μ̂_k)² (assign to nearest centroid) | p.18–19 |
| Mixture deviance | total within-cluster SS (sklearn `inertia_`) | p.18, p.22, p.30 |
| Standardization | x̃_ij = (x_ij − x̄_j)/sd(x_j) | p.23 |
| IC for K-means | D = −2 log LHD = within-cluster SS; df = K × p; "usual AIC/BIC" (L4: D + 2df, D + log(n) df) | p.30 |
| Implied K-means model | p_k(x) = Π_j N(x_j \| μ_kj, σ²) | p.32 |
| kNN graph | W_ij = 1 if i ∈ kNN(j) or j ∈ kNN(i), else 0 | p.38 |
| Gaussian kernel graph | W_ij = exp(−d²(i,j)/c) | p.38 |
| Degree matrix | G_ii = Σ_j W_ij | p.39 |
| Graph Laplacian | L = G − W; L𝟏 = 0; v⊤Lv = ½ Σ_ij W_ij (v_i − v_j)² ≥ 0 | p.39 |
| Zero eigenvalues | # zero eigenvalues of L = # connected pieces; eigenvectors are piece indicators | p.40 |
| Spectral embedding | V_{n×K} = K eigenvectors of L with smallest eigenvalues; K-means on rows of V | p.41 |
| Legislator kernel | W_ij = exp(−d_ij/669), d_ij = # roll calls on which i, j differ | p.46 |

## 10. Code idioms (house style, verbatim)

```python
# GMM (p.13)
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3)
gmm.fit(X)
gmm_clusters = gmm.predict(X)
# attributes: gmm.weights_, gmm.means_, gmm.covariances_

# K-means (p.22)
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3,init='random',n_init=1,max_iter=10)
kmeans.fit(X)
np.around(kmeans.cluster_centers_,2)
from collections import Counter
Counter(kmeans.labels_)
round(kmeans.inertia_,2)

# Spectral (p.41) - import not shown on slide (sklearn.cluster)
SpectralClustering(n_clusters=K, affinity='nearest_neighbors')
```
- Pandas: the p.47 table is a DataFrame with columns `Name`, `second_eig` and `Group`, sorted by `second_eig`.
- The course shows no hierarchical-clustering code. The dendrogram was probably made with scipy (see p.51 note).

## 11. Course conventions established in L7

| Convention | Slide |
|---|---|
| Unsupervised = models for x alone; the goal is still "low deviance, without overfit". | p.3 |
| Clustering ≠ classification. If group labels are known, it is classification. | p.4 |
| Clustering is primarily **exploratory/descriptive**. | p.5, p.19, p.29 |
| K-means needs **numeric/continuous** x and does worse on dummies/counts. | p.15, p.22, p.32 |
| **Standardize each feature before K-means**: (x − x̄)/sd. | p.23 |
| Use **multiple random starts** and keep the run with the lowest within-cluster SS (mixture deviance). | p.19, p.21 |
| K-means deviance = within-cluster SS; df = K·p; AIC/BIC as usual but "lower quality" than in regression. | p.18, p.30 |
| **Choose K by interpretability first**; IC-chosen K (MNIST: BIC 30, AIC 130) is "way more complicated than is useful". | p.19, p.29–31 |
| Cluster-to-classifier evaluation: majority label per cluster, accuracy on the train set **and** a held-out test set. | p.25, p.31 |
| Spectral: count near-zero Laplacian eigenvalues (the eigengap) to pick K. | p.42, p.46 |
| Hierarchical: agglomerative; cut the dendrogram horizontally for a partition. | p.49, p.51 |
| The similarity/distance measure determines the clustering. | p.8, p.48, p.52 |

## 12. Pitfalls and warnings emphasized

1. "Clustering should not be confused with classification!" (p.4)
2. The GMM likelihood is multimodal and "may stuck at local minimums" (p.12).
3. The K-means "solution depends on start location. Try multiple, take the best answer." (p.19, p.21)
4. "Scale matters": an unscaled feature with a larger range dominates cluster membership (p.23, p.32).
5. K-means implicitly assumes independence across dimensions and equal variance (p.32). It fails on:
   - non-convex shapes (rings, moons; p.33–34);
   - unequal variances (p.35);
   - anisotropic or correlated data (p.36).
6. K-means is worse for dummies or counts (p.32).
7. There is no ground truth, so CV for choosing K is problematic because k_i is latent (p.29).
8. AIC/BIC assumptions are "only roughly true for K-means" and the tools are "lower quality here than in regression".
   IC-chosen K can be far too large (p.30–31).
9. Over-interpreting clusters (the xkcd cartoon, p.28). Clusters can be impure: the MNIST "1" cluster is only 5/10
   ones (p.27).
10. Cluster labels are arbitrary and permute across runs and methods (p.14, p.43, p.44). This is visual, not stated.
11. With a fully connected graph, the first eigenvalue is 0 and its eigenvector is constant, which "says nothing".
    Use the second eigenvector (p.46).
12. The choice of similarity measure determines the result (p.8, p.52).

## 13. Datasets and worked examples

| Example | Slides | Key numbers |
|---|---|---|
| Trader clusters (Aït-Sahalia & Brunetti 2020, JoE) | p.7 | HFT, fundamental buyers/sellers, residual traders; Aug 2010 |
| 1-D 4-component mixture density | p.10–11 | modes ≈ 10, 20, 23, 33 |
| GMM on 3 clusters with unequal spread | p.14 | GMM recovers all three |
| K-means toy (n = 300, K = 3) | p.20, p.22 | WCV 70.1 → 65.72 → 55.33 → 24.44; centers (0, .03), (0, .99), (1.01, .97); counts 101/105/94 |
| K-means multiple runs (n = 250, K = 4) | p.21 | WCV 22.18 / **16.13** / 18.23 |
| MNIST (60k × 784 train, 10k × 784 test) | p.24–27, p.31 | K = 10: 57.84% train, 59.45% test; K = 30: 75.085% train, 76.29% test; BIC → 30, AIC → 130 |
| Failure shapes: rings, moons, unequal variances, anisotropic | p.33–36, p.43–44 | K-means gets 51% on rings; spectral clustering is exact |
| Six-point two-triangle graph | p.39–40 | eigenvalues 0, 0, 3, 3, 3, 3 |
| Two rings, 600 points, 10-NN | p.42 | eigenvalues 0, 0, 0.031, 0.036 |
| House roll calls (434 members × 669 votes) | p.45–47 | eigenvalues 0, 0.66, 0.98, 0.98; sign of the 2nd eigenvector = party for all 433 D/R |
| 7-point agglomerative example and dendrogram | p.50–51 | merge heights 0.14, 0.22, 0.35, 0.43, 0.78, 1.08 |
| WSJ news topics (Bybee, Kelly, Manela & Xiu 2024, JF) | p.53 | 180 topics × 18,433 |

---

## 14. Mapping to the final exam (Final_Autumn_2026-1.ipynb)

**Summary.** L7 is **peripheral** to Problems 1 and 2. For Problem 3 it offers optional exploratory and
feature-engineering tools. **No exam step requires a clustering method.** Any clustering used in Problem 3 must obey
the notebook's rule that "every scaler, penalty and hyper-parameter is chosen on training data only". The same logic
applies to cluster centroids and memberships: a clustering fitted on the full sample and then used as a feature is
a transformation fitted before the split, which L5 p.58 calls FATAL.

### Problem 1 (trees and ensembles, Social_Network_Ads)
- **p.4 (clustering vs classification):** Problem 1 is **supervised classification**, because the labels
  (`Purchased`) are known. Clustering is not the right tool, and L7 says so. Trees, RF and GB come from L8, the
  baseline and accuracy from L6/L8.
- **1.1 majority-class baseline:**
  - L7's majority-label idea (p.25, p.31) predicts each cluster's majority label. The "nobody purchases" baseline is
    the degenerate one-group version: predict the overall majority class.
  - This is a conceptual parallel only. The baseline itself should be cited from L6/L8.
- **1.1 max_leaf_nodes sweep, ties go to the smaller tree:** there is a loose parallel with p.29–31, where the
  simpler, more interpretable model is preferred when a more complex one ("way more complicated than is useful")
  adds little. This is analogy only; the tie rule itself is set by the exam.
- **1.1 and 1.3 (randomness):** p.19 and p.21 ("solution depends on start location") motivate fixing the seed. The
  exam sets `random_state=7034` for the folds.
- **Scaling:** p.23 says scale matters for **distance-based** methods. Trees split on thresholds, so they are
  invariant to monotone rescaling. Standardizing `Age` and `EstimatedSalary` is therefore unnecessary for Problem 1.
  L7 says only that scale matters for K-means; the conclusion about trees is my inference.
- **1.5:** L7 has nothing on time-series CV.

### Problem 2 (VaR pipeline trained on its own output)
- L7 has **essentially nothing** directly applicable.
- **p.3 framing:** the pipeline is an unsupervised "model for x alone". It fits a single Gaussian N(0, σ²) to x,
  which is the K = 1 case of the p.12 GMM.
- **2.4(b) loss of information:** real dj30 returns have excess kurtosis, and a single normal cannot represent it.
  - L7 p.10–12 shows that a mixture of Gaussians with different variances has a marginal density that a single
    Gaussian cannot match (multiple modes, different spreads).
  - Using this to explain fat tails is my extension. L7 never discusses kurtosis.
  - Do **not** change the pipeline the exam specifies.
- **p.30 (D = −2 log LHD):** this is the deviance/likelihood link, if needed for wording.

### Problem 3 (research project: 50 assets × 300 months)
- **"Know your data": correlation matrix ordered by class, block structure, pooling.**
  - Spectral clustering (p.38–46) or hierarchical clustering (p.48–52) on an asset-similarity matrix built from
    **training-window** returns can show whether data-driven groups recover the four given classes. It is the same
    logic as p.46, where the sign of the 2nd eigenvector recovered party.
  - For example, use a fully connected graph with W_ij = exp(−d²(i,j)/c), where d is based on correlation distance.
  - Following p.5, this is **exploratory** evidence on whether pooling across classes is sensible.
  - Per p.4, since asset classes are **given**, per-class vs pooled modelling uses known labels. Clustering would
    only be for discovering structure beyond the labels, such as groups of countries by macro behaviour.
- **Feature engineering on the extended macro file (x17–x164, duplicates, uneven quality).**
  - Hierarchical clustering of the macro series on a correlation-based distance (p.48–53) can find exact duplicates
    (they merge at height ≈ 0) and group near-duplicates into themes. This parallels the WSJ topic tree on p.53.
  - Pick one representative per branch, or average within a branch, to reduce dimension. That is the "summary (DR)"
    use on p.4.
  - Because p.48 and p.52 do not specify a linkage, state the one you choose and justify it.
  - Fit on the training window only. The exam's duplicate check ("find them before you concatenate") can also be
    done directly with equality or correlation tests; clustering is optional.
- **Standardization (p.23).** The z-score (x − x̄)/sd(x) is the same formula the exam uses for cross-sectional
  standardization of characteristics within month.
  - In L7 it is applied column-wise over the sample. For Problem 3, apply it **across assets within each month**
    (no look-ahead, since only same-date data is used).
  - For global macro, use **trailing** moments only, per the notebook.
  - The p.23 message, that unscaled features dominate distance-based methods, also supports "scales differ
    enormously by class" and the **vol-scaled target** r/σ̂_{t−1}.
  - p.35 (unequal variances break K-means) is a visual analogue of pooling assets with very different volatilities.
    That parallel is my inference.
- **Regime features.**
  - K-means or GMM (p.12–22) on lagged, trailing-standardized global macro can define "macro states" to interact
    with characteristics. The exam suggests "interactions between characteristics and macro states".
  - Choose K descriptively (p.19, p.29), with BIC as secondary support (p.30). Fit only on data available up to
    each refit date (expanding window).
  - Use multiple starts (p.21, `n_init` > 1).
  - Do **not** cluster on dummies (p.32).
  - Cluster labels are arbitrary and can **relabel between refits**, so map them consistently, for example by
    ordering clusters on their centroid values.
  - Treat such features as exploratory. Test them with the same R²_OOS and placebo design as the other macro
    predictors.
- **3-portfolio.** p.6 ("identify groups of stocks with similar characteristics to create portfolios that try to
  profit from differences within/across groups") motivates cluster- or sort-based long-short portfolios.
  - Tercile sorts on characteristics, suggested in the notebook, are the simplest "grouping" version. L7 does not
    teach portfolio construction.
- **3-evaluation.** L7 provides no R²_OOS, Sharpe ratio or window design. Those come from L4/L5 and the notebook.
  If clustering is used as a prediction aid, p.25 and p.31 show the house practice of reporting both in-sample
  (train) and held-out (test) performance.

---

## 15. Boundary checklist: mentioned but NOT taught in L7
- Factor models: named on p.3 only.
- EM algorithm: named on p.13 only, not derived.
- Divisive hierarchical clustering: defined on p.49, then set aside.
- Linkage types (single/complete/average/Ward): never named. "Different ways to measure similarity" (p.48, p.52).
- Normalized Laplacian: mentioned only as what sklearn uses (p.41), not derived.
- Collaborative filtering: named only (p.5).
- The trader clustering method (p.7) and the WSJ topic model or clustering method (p.53): figures only.
- Cross-validation for choosing K: raised as problematic (p.29). No procedure is given.
- Not present at all: silhouette/elbow/gap statistics, DBSCAN, K-medoids, cluster stability, time-series issues,
  scaling code (StandardScaler), and GMM `covariance_type`/`bic`/`predict_proba`.
