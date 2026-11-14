# Qwen3.8 single-factor comparison

Baseline: `20261114T054137Z-0e00a3ea0c`; candidate: `20261114T054955Z-b62d4eb4a9`.

GPU layers: 24 → 40. Quality passed on the fixed fixtures in both runs.

| Task | Baseline mean s | Candidate mean s | Observed latency ratio | Output tokens baseline → candidate |
|---|---:|---:|---:|---|
| summary-pilot | 13.99 | 9.43 | 1.48× | [39, 39] → [39, 39] |
| summary-incident | 13.58 | 9.21 | 1.47× | [38, 38] → [38, 38] |
| json-person | 10.27 | 6.75 | 1.52× | [27, 27] → [27, 27] |
| json-orders | 12.87 | 8.32 | 1.55× | [34, 34] → [34, 34] |
| code-default-list | 45.44 | 31.92 | 1.42× | [138, 138] → [138, 138] |
| code-missing-last | 49.20 | 34.08 | 1.44× | [149, 149] → [150, 150] |

Measured request totals: 290.67s → 199.39s (1.46× observed latency ratio).

Resource trade-offs (sampled peaks, bytes):
- gpu_device_used_bytes: {'status': 'ok', 'sampled_peak_bytes': 7403274240} → {'status': 'ok', 'sampled_peak_bytes': 11142496256}
- system_memory_used_bytes: {'status': 'ok', 'sampled_peak_bytes': 19506254848} → {'status': 'ok', 'sampled_peak_bytes': 19351221248}
- server_rss_bytes: {'status': 'ok', 'sampled_peak_bytes': 17728733184} → {'status': 'ok', 'sampled_peak_bytes': 17728950272}

Limits:
- Two repeats only; no statistical significance or production SLA claim.
- Host load may differ; sequential runs are not randomized.
- Output lengths/content may differ despite fixed prompts and generation settings; inspect per-case token counts.
- GPU/system peaks include competing processes; RSS is owned server only; all peaks sampled.
- New process, warm filesystem cache not controlled; TTFT unavailable from non-stream requests.
