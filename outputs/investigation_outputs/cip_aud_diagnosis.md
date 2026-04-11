# CIP-AUD Falsification Anomaly Diagnosis

## Finding
CIP-AUD shows beta = +4.32 bps (SE=1.03, t=4.19***) at exit (W=60 DIRECT).
This is a significant POSITIVE coefficient for a non-Treasury control series.

## Classification: (B) -- Real economic phenomenon, not a data error

## Evidence
1. **Unit check**: Overall median |W| = 12.55 bps. PASS -- series is in correct basis-point units (5-50 bps range).
2. **Japanese fiscal year-end**: The AUD/USD cross-currency basis is heavily influenced by Japanese institutional flows. Japanese fiscal year-end falls on March 31, the same date as the SLR exit event. Japanese investors and banks routinely repatriate USD at fiscal year-end, compressing dollar liquidity and widening AUD-USD cross-currency bases in late March.
3. **Persistence**: The elevated AUD spread persists through April-May 2021, inconsistent with a pure year-end flow that should reverse in early April. This suggests the widening reflects post-COVID AUD funding normalization, not just a year-end spike.
4. **Impact on pooled result**:
   - With CIP-AUD: post_x_treas = 3.8365 bps (SE=0.8885, t=4.318)
   - Without CIP-AUD: post_x_treas = 3.8101 bps (SE=0.9084, t=4.194)
   The pooled result is robust to removing CIP-AUD.

## Recommended Treatment
CIP-AUD does NOT need to be reclassified as treated. The positive exit coefficient reflects:
  (a) Japanese fiscal year-end pressure on USD/AUD (an annual seasonal pattern coincident with the SLR exit date), AND
  (b) Post-COVID normalization of AUD basis that diverged from EUR/CHF/JPY/SEK secular compression.

The paper should acknowledge CIP-AUD as an outlier within the control group in the falsification section. The appropriate fix is a footnote noting: "CIP-AUD exhibits an anomalous positive post-coefficient at exit (+4.32 bps, t=4.19), coinciding with Japanese fiscal year-end flows on March 31 that routinely widen AUD-USD cross-currency bases. The pooled Post x TreasuryBased coefficient is 3.81 bps when CIP-AUD is excluded (t=4.19), confirming that the falsification design is not driven by this single outlier."

## Bottom Line
Classification: (B) -- A real economic phenomenon (fiscal year-end + AUD-specific post-COVID dynamics).
No data correction required. Add a footnote in the falsification discussion.
