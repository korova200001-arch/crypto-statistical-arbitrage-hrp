# crypto-statistical-arbitrage-hrp
Market-Neutral Statistical Arbitrage &amp; HRP Portfolio Optimization Terminal


================================================================================
FINANCIAL FINTECH PRODUCT: MARKET-NEUTRAL STATISTICAL ARBITRAGE TERMINAL (HRP)
================================================================================
Author: Master of Science in Financial Markets, HSE University (2026)
Domain: Quantitative Trading / Econometric Risk Management
Tech Stack: Python, Streamlit, CCXT, Scipy, Statsmodels

1. PROJECT OVERVIEW
--------------------------------------------------------------------------------
This Fintech Minimum Viable Product (MVP) is an interactive, web-based 
Quantitative Trading Terminal designed for real-time asset selection, systemic 
risk clustering, and market-neutral backtesting. 

Unlike traditional heuristic trading bots, this terminal leverages advanced 
econometric filters to exploit asset price inefficiencies while enforcing 
strict volatility-scaled risk allocation.

2. QUANTITATIVE FRAMEWORK & MATHEMATICAL CORE
--------------------------------------------------------------------------------
The core infrastructure is divided into three distinct quantitative pipelines:

A. The Covariance Risk Matrix Engine
   Instead of using simple Pearson linear correlation—which is prone to structural 
   breaks and regime shifts—the engine computes a localized covariance matrix 
   of asset returns annualized for cryptocurrency markets:
   
   Σ = Cov(R_i, R_j) * 365

   Covariance captures both the joint directional co-movement and the absolute 
   scale of historical volatility (variance). This prevents the algorithm from 
   entering "paper arbitrage" traps where highly correlated assets diverge due 
   to highly asymmetric intrinsic variances.

B. Hierarchical Risk Parity (HRP) & Quasi-Diagonalization
   To solve the historical instability challenges of Markowitz's mean-variance 
   optimization (the inversion of large, noisy covariance matrices), the system 
   implements Marcos López de Prado's Hierarchical Risk Parity (HRP) logic:
   1. Distance Metric Formulation: A distance matrix is computed using 
      D(i, j) = sqrt(0.5 * (1 - P(i, j))), where P is the correlation matrix.
   2. Agglomerative Hierarchical Clustering: Assets are clustered using 
      Ward's linkage method to build an economic tree (dendrogram).
   3. Quasi-Diagonalization: The covariance matrix is reordered so that 
      highly co-dependent assets are placed along the main diagonal, visually 
      segmenting the crypto market into isolated sector-specific risk blocks 
      (e.g., AI clusters, heavy beta clusters, L1/L2 networks).

C. Market-Neutral Mean-Reversion Backtester
   Once an optimal stationary pair (e.g., AR/USDT vs. ARKM/USDT) is identified:
   1. Hedge Ratio Computation: Computed dynamically to balance the dollar-exposure 
      of both legs, ensuring the portfolio beta is strictly managed.
   2. Synthetic Spread Formation: Spread_t = Price_A,t - (Hedge_Ratio * Price_B,t)
   3. Rolling Z-Score Normalized Signal: 
      Z_t = (Spread_t - Mean_rolling) / Std_rolling
   4. Execution Strategy: 
      - Short Spread (Short Asset A, Long Asset B) when Z_t > Entry_Threshold.
      - Long Spread (Long Asset A, Short Asset B) when Z_t < -Entry_Threshold.
      - Exit position when Z_t converges back to its mean (0.0).

3. HOW TO PRESENT THIS MVP TO PROFESSORS / FUNDS
--------------------------------------------------------------------------------
When demonstrating this screen on your computer, frame the discussion around 
these core competitive pillars:

- "The system does not predict price direction; it captures Alpha by exploiting 
   short-term mathematical dislocations within structurally tied economic clusters."
- "The HRP Quasi-Diagonalization tab demonstrates how a fund can allocate capital 
   without suffering from the mathematical instabilities of traditional modern 
   portfolio theory (MPT)."
- "The terminal includes an adjustable leverage module and rolling lookback 
   windows, enabling dynamic out-of-sample backtesting under volatile market regimes."

4. LOCAL RUN INSTRUCTIONS
--------------------------------------------------------------------------------
To boot the interactive terminal locally, execute via Windows PowerShell:
> python -m streamlit run mvp.py

The environment will automatically open the local host application at:
URL: http://localhost:8501
