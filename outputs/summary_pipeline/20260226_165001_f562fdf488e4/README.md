# Summary pipeline run

## Config

```json
{
  "event_dates": [
    "2020-04-01",
    "2021-03-19",
    "2021-03-31"
  ],
  "windows": [
    3,
    5,
    10
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
  "dependent_series": null,
  "tenor_subset": null,
  "total_controls": [],
  "direct_controls": [
    "sofr",
    "tgcr",
    "bgcr"
  ],
  "hac_lags": 5,
  "bootstrap_reps": 200,
  "bootstrap_block_size": 5,
  "random_seed": 42,
  "output_root": "outputs/summary_pipeline",
  "cache_root": "outputs/cache"
}
```

## Notes

Processed 14 daily_long rows across 1 tenors and 2 series.
