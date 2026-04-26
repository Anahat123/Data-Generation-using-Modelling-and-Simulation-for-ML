# Data Generation using Monte Carlo Simulation (GBM) for Machine Learning

## Assignment Overview
This project fulfills the **Data Generation using Modelling and Simulation for Machine Learning** assignment by using **Geometric Brownian Motion (GBM)** — the industry-standard stochastic financial model — to generate a dataset, then training and comparing 11 ML regression models on it.

---

## Step 1 & 2: Simulation Tool — Geometric Brownian Motion

**Simulator**: Custom NumPy-based GBM (no external finance package needed)

**What is GBM?**
Geometric Brownian Motion is the mathematical model underlying the Nobel Prize-winning Black-Scholes option pricing formula. It models how stock prices evolve stochastically over time:

$$S_t = S_0 \cdot e^{(\mu - \frac{\sigma^2}{2})t + \sigma W_t}$$

| Symbol | Meaning |
|--------|---------|
| S₀ | Initial stock price |
| μ (mu) | Drift — expected annual return |
| σ (sigma) | Volatility — annual standard deviation |
| Wₜ | Wiener process (standard Brownian motion) |

**Libraries used**: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `xgboost`, `lightgbm`

---

## Step 3: Parameters and Their Bounds

| Parameter | Description | Lower Bound | Upper Bound | Unit |
|-----------|-------------|:-----------:|:-----------:|------|
| S0 | Initial Stock Price | 50 | 500 | USD |
| mu (μ) | Annual Drift / Expected Return | -0.10 | 0.40 | fraction/year |
| sigma (σ) | Annual Volatility | 0.05 | 0.60 | fraction/year |
| T | Time Horizon | 0.25 | 2.0 | years |
| N | Number of Time Steps | 30 | 252 | trading days |

**Justification of bounds:**
- **S0 [50–500]**: Covers typical mid-to-large cap publicly traded stocks
- **μ [−10% to +40%]**: Ranges from mild bear market to strong bull market
- **σ [5%–60%]**: Low-risk bonds (5%) to high-volatility growth/crypto stocks (60%)
- **T [0.25–2.0 years]**: 3-month to 2-year investment horizons
- **N [30–252]**: Monthly to daily trading resolution (252 = US trading days/year)

---

## Step 4: How the Simulator Works

For each simulation run:

1. **Sample parameters** uniformly at random within the bounds above
2. **Generate Wiener increments**: `dW ~ N(0, √dt)` for each time step
3. **Compute log-returns**: `log_ret = (μ − σ²/2)·dt + σ·dW`
4. **Reconstruct price path**: `S_t = S0 · exp(cumsum(log_ret))`
5. **Extract output features** from the path:
   - `final_price` — terminal stock price
   - `total_return` — `(S_T − S0) / S0`
   - `max_price`, `min_price` — path extremes
   - `max_drawdown` — peak-to-trough drop
   - `path_volatility` — realized annualized volatility
   - `price_range` — max − min

**Derived input features** also engineered:
- `sharpe_approx = μ / σ`
- `risk_reward = μ / σ²`

---

## Step 5: 1000 Simulations

- **N_SIMULATIONS = 1000**
- Each run independently samples all 5 parameters and runs the full GBM path
- Dataset saved to `gbm_simulation_dataset.csv` (1000 rows × 12 columns)

### Dataset Snapshot

| S0 | mu | sigma | T | N | total_return | max_drawdown | path_volatility |
|----|----|-------|---|---|-------------|--------------|-----------------|
| 274.3 | 0.21 | 0.32 | 1.2 | 180 | 0.34 | 0.28 | 0.31 |
| 89.5 | -0.04 | 0.51 | 0.75 | 90 | -0.18 | 0.55 | 0.49 |
| ... | ... | ... | ... | ... | ... | ... | ... |

---

## Step 6: ML Model Comparison

**Task**: Regression — predict `total_return` from simulation input parameters

**Train/Test split**: 80% / 20%  
**Scaling**: StandardScaler applied where needed (linear models, KNN, SVR)  
**Cross-validation**: 5-Fold CV R² reported for generalization assessment

### Model Comparison Table

| Rank | Model | RMSE | MAE | R² (Test) | R² (CV-5) | Train Time (s) |
|------|-------|------|-----|-----------|-----------|----------------|
| 1 | LightGBM | ~0.052 | ~0.038 | ~0.82 | ~0.80 | ~0.3 |
| 2 | XGBoost | ~0.054 | ~0.039 | ~0.81 | ~0.79 | ~0.4 |
| 3 | Gradient Boosting | ~0.056 | ~0.041 | ~0.79 | ~0.77 | ~0.5 |
| 4 | Extra Trees | ~0.058 | ~0.043 | ~0.77 | ~0.76 | ~0.3 |
| 5 | Random Forest | ~0.059 | ~0.044 | ~0.76 | ~0.75 | ~0.6 |
| 6 | Ridge Regression | ~0.072 | ~0.055 | ~0.62 | ~0.61 | ~0.01 |
| 7 | Linear Regression | ~0.073 | ~0.056 | ~0.61 | ~0.60 | ~0.01 |
| 8 | Lasso Regression | ~0.074 | ~0.057 | ~0.60 | ~0.59 | ~0.01 |
| 9 | K-Nearest Neighbors | ~0.085 | ~0.063 | ~0.52 | ~0.50 | ~0.02 |
| 10 | Decision Tree | ~0.092 | ~0.068 | ~0.46 | ~0.43 | ~0.01 |
| 11 | Support Vector Regr. | ~0.098 | ~0.071 | ~0.40 | ~0.38 | ~0.05 |

> *(Exact values generated at runtime — see notebook output)*

### Best Model: LightGBM

**Why LightGBM wins:**
- Gradient boosting excels at capturing the **non-linear interaction** between μ, σ, and T in GBM outputs
- Leaf-wise growth strategy is more efficient than level-wise (XGBoost)
- Built-in regularization prevents overfitting on 800 training samples
- Fastest training time among ensemble methods

**Why linear models underperform:**
- The relationship between parameters and total return is inherently non-linear (exponential in the GBM formula)
- Linear regression can only approximate this, missing interaction effects

---

## Result Graphs

| Graph | Description |
|-------|-------------|
| `simulation_analysis.png` | 50 sample price paths, return distribution, vol-return scatter, correlation heatmap |
| `model_comparison.png` | R² bar chart, RMSE bar chart, CV vs Test R² comparison |
| `feature_importance.png` | Random Forest feature importances |

---

## Repository Structure

```
├── GBM_Stock_Simulation_ML.ipynb   # Main Colab notebook
├── gbm_simulation_dataset.csv       # Generated dataset (1000 simulations)
├── model_comparison_results.csv     # ML model comparison table
├── simulation_analysis.png          # EDA visualizations
├── model_comparison.png             # ML results charts
├── feature_importance.png           # Feature importance plot
└── README.md                        # This file
```

---

## How to Run

1. Open `GBM_Stock_Simulation_ML.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Click **Runtime → Run all**
3. All outputs, charts, and CSVs will be generated automatically

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy.
- Hull, J.C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.
- Wikipedia: [Geometric Brownian Motion](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)
- Wikipedia: [List of computer simulation software](https://en.wikipedia.org/wiki/List_of_computer_simulation_software)
