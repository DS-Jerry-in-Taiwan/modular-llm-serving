# Qwen3.8 single-factor comparison

Baseline: `20261114T131732Z-1327935c51`; candidate: `20261114T132436Z-34b235865f`.

GPU layers: 24 → 40. Quality passed on the fixed fixtures in both runs.

| Task | Baseline mean s | Candidate mean s | Observed latency ratio | Output tokens baseline → candidate |
|---|---:|---:|---:|---|
| summary-pilot | 18.93 | 10.13 | 1.87× | [39, 39] → [39, 39] |
| summary-incident | 16.89 | 10.03 | 1.68× | [38, 38] → [38, 38] |
| json-person | 13.58 | 7.48 | 1.81× | [27, 27] → [27, 27] |
| json-orders | 14.26 | 9.10 | 1.57× | [34, 34] → [34, 34] |
| code-default-list | 50.58 | 34.60 | 1.46× | [138, 138] → [138, 138] |
| code-missing-last | 57.30 | 37.69 | 1.52× | [149, 149] → [150, 150] |

Measured request totals: 343.08s → 218.10s (1.57× observed latency ratio).

Resource trade-offs (sampled peaks, bytes):
- gpu_device_used_bytes: {'status': 'ok', 'sampled_peak_bytes': 7396524032} → {'status': 'ok', 'sampled_peak_bytes': 11134042112}
- system_memory_used_bytes: {'status': 'ok', 'sampled_peak_bytes': 21982096384} → {'status': 'ok', 'sampled_peak_bytes': 22361662464}
- server_rss_bytes: {'status': 'ok', 'sampled_peak_bytes': 17722429440} → {'status': 'ok', 'sampled_peak_bytes': 17728630784}

Interpretation: these short Chinese summary, structured extraction and Python explanation/repair fixtures passed independent quality review. They support these limited uses on this configuration, not full-agent coding, long documents, image/video/audio generation or a general capability guarantee.

Limits:
- Two repeats only; no statistical significance or production SLA claim.
- Host load may differ; sequential runs are not randomized.
- Output lengths/content may differ despite fixed prompts and generation settings; inspect per-case token counts.
- GPU/system peaks include competing processes; RSS is owned server only; all peaks sampled.
- New process, warm filesystem cache not controlled; TTFT unavailable from non-stream requests.
