# summary_pipeline run

## Config
```json
{
  "outcomes_source": "tips_treasury_implied_rf_2010",
  "outcome_pattern": "arb_",
  "tenors_required": [
    2,
    5,
    10
  ],
  "events": [
    "2020-04-01",
    "2021-03-19",
    "2021-03-31"
  ],
  "windows": [
    20,
    60
  ],
  "event_bins": [
    [
      -60,
      -41
    ],
    [
      -40,
      -21
    ],
    [
      -20,
      -1
    ],
    [
      0,
      0
    ],
    [
      1,
      20
    ],
    [
      21,
      40
    ],
    [
      41,
      60
    ]
  ],
  "total_controls": [
    "VIX",
    "HY_OAS",
    "BAA10Y",
    "issu_7_bil",
    "issu_14_bil",
    "issu_30_bil"
  ],
  "direct_controls": [
    "VIX",
    "HY_OAS",
    "BAA10Y",
    "issu_7_bil",
    "issu_14_bil",
    "issu_30_bil",
    "SOFR",
    "spr_tgcr",
    "spr_effr"
  ],
  "hac_lags": 5,
  "run_layer2": true
}
```

## Notes
```
No main dataframe found (looked for: analysis_panel, arb_panel, panel, daily_long, pivot).
```