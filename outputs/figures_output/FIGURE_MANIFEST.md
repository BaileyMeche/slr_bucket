# Figure Manifest
## Figure 1 — Time Series Overview
**Files:** `fig1_series_overview.pdf`, `fig1_series_overview.png`

**Data sources:**
- data/series/cip_spreads_3m_bps.csv
- data/series/treasury_sf_output.csv
- data/series/tips_treasury_implied_rf_2010.parquet
- data/series/equity_spot_spread_SPX.csv
- data/series/equity_spot_spread_NDX.csv
- data/series/equity_spot_spread_INDU.csv

**LaTeX:**
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{figures_output/fig1_series_overview.pdf}
  \caption{...}
  \label{fig:series_overview}
\end{figure}
```

## Figure 2 — UST SF Detail by Tenor
**Files:** `fig2_ust_sf_detail.pdf`, `fig2_ust_sf_detail.png`

**Data sources:**
- data/series/treasury_sf_output.csv

**LaTeX:**
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{figures_output/fig2_ust_sf_detail.pdf}
  \caption{...}
  \label{fig:ust_sf_detail}
\end{figure}
```

## Figure 3 — Exit Event Window Paths
**Files:** `fig3_exit_event_paths.pdf`, `fig3_exit_event_paths.png`

**Data sources:**
- Panel data from all series

**LaTeX:**
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{figures_output/fig3_exit_event_paths.pdf}
  \caption{...}
  \label{fig:exit_event_paths}
\end{figure}
```

## Figure 4 — Coefficient Plot (Forest Plot)
**Files:** `fig4_coef_plot.pdf`, `fig4_coef_plot.png`

**Data sources:**
- audit_repairs/outputs/jump_estimates_pooled.csv
- audit_repairs/outputs/did_clean_baseline.csv

**LaTeX:**
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{figures_output/fig4_coef_plot.pdf}
  \caption{...}
  \label{fig:coef_plot}
\end{figure}
```

## Figure 5 — Series-Level Coefficients (Exit W=60 DIRECT)
**Files:** `fig5_series_level_coefs.pdf`, `fig5_series_level_coefs.png`

**Data sources:**
- audit_repairs/outputs/jump_estimates_by_series.csv

**LaTeX:**
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{figures_output/fig5_series_level_coefs.pdf}
  \caption{...}
  \label{fig:series_level_coefs}
\end{figure}
```

## Figure 6 — Regime Box Plots
**Files:** `fig6_regime_boxplots.pdf`, `fig6_regime_boxplots.png`

**Data sources:**
- Panel data from all series

**LaTeX:**
```latex
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{figures_output/fig6_regime_boxplots.pdf}
  \caption{...}
  \label{fig:regime_boxplots}
\end{figure}
```

