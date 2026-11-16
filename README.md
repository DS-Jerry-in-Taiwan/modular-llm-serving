# Modular LLM Serving：本機推論與可復用服務

## 1. 專案簡介

本專案提供一個可重跑的本機 LLM inference lab，以及一個可持續開發的 resident inference service。主要驗證環境是單機、單張 GPU、Qwen3.8-27B GGUF 與 pinned llama.cpp runtime。

### 核心能力

- **模型與 runtime 驗證**：檢查模型檔、runtime revision、GPU 狀態與 pinned configuration。
- **可重現效能實驗**：以固定資料集比較不同 GPU offload 設定，保存 metrics、resource samples 與 provenance。
- **Resident serving**：以單一常駐 engine 提供多次請求，包含 admission、token/context budget、idempotency、cancellation、deadline 與 streaming。
- **Durable job state**：將請求生命週期寫入 SQLite，並在重啟時對未完成工作採取明確 recovery policy。
- **Observability 與 LLMOps gate**：保存關聯事件、phase metrics、evidence digest 與 provenance，並以離線 gate 驗證可接受性。

### 執行模式

| 模式 | 目的 | 執行方式 | 介面 |
| --- | --- | --- | --- |
| V1 cold benchmark | 公平比較 24/40 GPU layers 與固定任務 | 每個工作建立、使用並清理自己的 `llama-server` | CLI、`/api/*` |
| V2 resident service | 持續處理多次推論請求 | 啟動一次 engine，由 admission queue 管理請求 | `/api/v2/*` |

兩種模式共用模型與 runtime identity 驗證，但使用不同的 request/response contract。Modal + Gemma smoke/deployment path 另行保留，不屬於本機 Qwen serving runtime。

### 目前版本與驗收

- `v2.0.0`：Serviceable Inference，包含 V2 serving、durable jobs、observability、LLMOps gate 與 Demo/API integration。
- `v1.0.0`：Local Inference Lab V1，包含 Qwen baseline、24/40 GPU layers comparison 與 cold Demo。
- V2 foundation：38/38 targeted tests。
- V2 Demo/API integration：7/7 tests。
- 完整回歸：182/182 tests；Python compilation、JavaScript syntax、diff checks 通過。

自動化測試驗證模組流程與失敗處理，不等同於目標硬體上的實體 GPU 效能驗證。Qwen3.8-27B 在 12GB GPU 上的 throughput、VRAM/RAM 使用量與可用 context size，仍須在實際 driver、runtime 與模型檔環境中測量。

### 版本與發布狀態

| 版本 | 狀態 | 說明 |
| --- | --- | --- |
| [`v2.0.0`](https://github.com/DS-Jerry-in-Taiwan/modular-llm-serving/releases/tag/v2.0.0) | 目前正式版本 | V2 resident service 與 LLMOps foundation |
| [`v1.0.0`](https://github.com/DS-Jerry-in-Taiwan/modular-llm-serving/releases/tag/v1.0.0) | 已發布、保留相容 | V1 local inference lab |
| Modal/Gemma path | 獨立 legacy path | Modal deployment validation |

V2 目前不包含 Kubernetes、autoscaling、multi-GPU、dynamic batching 或多模態生成。

詳細 release notes 請見 [GitHub Releases](https://github.com/DS-Jerry-in-Taiwan/modular-llm-serving/releases)。

## 2. 高階架構

```mermaid
flowchart LR
  User[使用者 / CLI / Browser]
  Demo[V1/V2 Loopback Demo]
  Eval[Evaluation & LLMOps]
  V1[V1 Cold Runner\n每次建立 llama-server]
  V2[V2 Resident Service\n單一常駐 engine]
  Queue[AdmissionQueue\nFIFO / budget / idempotency]
  Supervisor[ResidentSupervisor\nstart / ready / stop]
  Runtime[llama.cpp llama-server\nLocalServer adapter]
  Store[(SQLite DurableJobStore)]
  Obs[Observability\nJSONL events / metrics / redaction]
  Evidence[(Local Evidence\nmanifest / response / reports)]
  Modal[Modal + Gemma legacy path]

  User --> Demo
  User --> Eval
  Demo --> V1
  Demo --> V2
  Eval --> V1
  V1 --> Runtime
  V2 --> Queue --> Supervisor --> Runtime
  V2 --> Store
  V2 --> Obs
  V1 --> Evidence
  Eval --> Evidence
  Store --> Evidence
  Modal -. preserved separately .-> User
```

### 邊界怎麼看？

- **Browser / CLI** 是輸入端，不直接控制模型程序。
- **Demo/API** 負責 HTTP 邊界、請求格式、loopback 安全與 UI 狀態呈現。
- **V1** 追求可重現比較；每個 variant 都有獨立 process、資源取樣與清理證據。
- **V2** 追求服務化；`ResidentSupervisor` 管一個 engine，`AdmissionQueue` 控制請求何時進入 engine。
- **Evaluation/LLMOps** 讀取已完成的 evidence，不混入任意 live prompt 的品質宣稱。
- **Modal path** 是既有部署驗證契約，不能與本機 Qwen 服務混為同一個 runtime。

## 3. 流程循序圖

以下是 V2 resident 請求從提交到 terminal state 的主要循序：

```mermaid
sequenceDiagram
  participant B as Browser/Client
  participant H as web.server
  participant M as ResidentDemoManager
  participant D as DurableJobStore
  participant Q as AdmissionQueue
  participant S as ResidentSupervisor
  participant E as llama-server
  participant O as Observability

  B->>H: POST /api/v2/runs
  H->>M: 解析並驗證 V2 request
  M->>D: create_job(state=queued)
  M->>Q: submit(request, deadline)
  H-->>B: 202 request_id / run_id
  Q->>S: start or reuse resident engine
  S->>E: ready()
  Q->>M: runner(engine, request, cancellation)
  M->>D: transition(running)
  M->>O: queue_complete / generation_start
  M->>E: generate(prompt, max_new_tokens)
  E-->>M: response or stream chunks
  M->>O: stream events / phase metrics
  alt completed
    M->>D: write response evidence + digest
    M->>D: transition(completed)
  else cancelled / deadline / failed
    M->>D: transition(terminal error)
  end
  M->>O: terminal event
  B->>H: GET /api/v2/runs/{id}
  H-->>B: public status, response/events/error
```

### 重要的終態規則

1. durable queued state 必須先寫入，避免請求已執行但沒有 ledger。
2. `cleanup` 完成前不能發佈 terminal success。
3. cancellation、deadline、queue-full 與 inference failure 使用不同 error code。
4. public API 不回傳 prompt、credential、runtime path 或 raw provider error。
5. restart 不繼續未完成的 generation；recovery 會依 durable policy 將 incomplete job 收斂成明確狀態。

## 4. 專案模組架構與交互

### 目錄與責任

```text
inference_lab/
├── config.py                 # pinned config/schema validation
├── acquisition.py            # model artifact acquisition + hash verification
├── preflight.py              # read-only readiness checks
├── runtime.py                # owned llama-server process + bounded HTTP
├── telemetry.py              # GPU/host probes and response metrics
├── artifacts.py              # immutable evidence allocation and writes
├── cli.py / __main__.py      # CLI orchestration
├── evaluation/
│   ├── tasks.py              # fixed dataset, filters, review binding
│   ├── baseline.py           # repeated measurements and baseline report
│   └── comparison.py         # identity and single-factor comparison
├── serving/
│   ├── supervisor.py         # one resident engine lifecycle
│   ├── admission.py          # FIFO queue, budgets, deadlines, cancellation
│   ├── streaming.py          # ordered chunks and terminal event
│   └── observability.py      # correlated events, metrics, redaction
├── v2_contracts/             # request/response/event/error schemas
├── llmops/
│   ├── durable.py             # SQLite job ledger and recovery
│   ├── ci_gate.py             # offline acceptance gate
│   └── registry.py            # versioned provenance/lineage
└── web/
    ├── server.py              # /api/*, /api/v2/* and static serving
    ├── jobs.py                # DemoManager + ResidentDemoManager adapters
    └── static/                # browser UI and lifecycle display
```

### 交互關係

| 使用情境 | 主要入口 | 主要模組交互 | 產出 |
| --- | --- | --- | --- |
| V1 固定 benchmark | `python -m inference_lab baseline` | `cli → evaluation.baseline → runtime → telemetry/artifacts` | repeated metrics、report、review evidence |
| V1 Demo | `POST /api/runs` | `web.server → DemoManager → runtime.LocalServer` | variant runs、cleanup receipt |
| V2 resident request | `POST /api/v2/runs` | `web.server → ResidentDemoManager → AdmissionQueue → ResidentSupervisor → engine` | durable job、response/events、terminal state |
| V2 evaluation gate | CI/offline gate | `llmops.ci_gate → durable evidence/registry` | pass/fail gate、provenance report |
| Modal legacy smoke | `modal deploy modal_llama_smoke.py` | Modal image + pinned llama.cpp server | deployment validation evidence |

模組依賴方向由上而下：transport (`web`/CLI) 呼叫 orchestration，orchestration 呼叫 serving/evaluation，底層再共用 `config`、`runtime`、`artifacts` 與 telemetry。V2 不會透過 HTTP 重新接收 model path 或 runtime argument；模型與 runtime identity 由本機設定固定。

## 5. 案例說明

### 案例 A：比較 24 與 40 GPU layers

使用 V1 cold benchmark：

```bash
venv/bin/python -m inference_lab baseline \
  --config experiments/qwen38-27b-q4.json \
  --dataset experiments/tasks-v1.json \
  --binary ~/.cache/inference-lab/runtime/llama-server \
  --model ~/.cache/inference-lab/models/Qwen3.8-27B-Q4_K_M.gguf \
  --output-root docs/local-inference/baseline
venv/bin/python -m inference_lab baseline \
  --config experiments/qwen38-27b-q4-gpu40.json \
  --dataset experiments/tasks-v1.json \
  --binary ~/.cache/inference-lab/runtime/llama-server \
  --model ~/.cache/inference-lab/models/Qwen3.8-27B-Q4_K_M.gguf \
  --output-root docs/local-inference/optimized
```

每次 run 都會驗證 identity、建立自己的 llama-server、執行固定案例、取樣資源、清理 process group，再寫入不可覆寫的 evidence。這條路徑適合回答「哪個設定比較快、耗多少 GPU/RAM」。

### 案例 B：啟動 V2 resident Demo

```bash
venv/bin/python -m inference_lab.web.server --execution-mode resident \
  --state-path docs/local-inference/v2-jobs.sqlite3
```

瀏覽 `http://127.0.0.1:7860/`，選擇 **Warm resident**，送出提示詞。第一次請求會啟動 engine，後續請求重用同一個 engine；UI 會顯示 request/run ID、queue、stream、terminal state 與 metrics。

### 案例 C：V2 API request

```bash
curl -sS -X POST http://127.0.0.1:7860/api/v2/runs \
  -H 'Content-Type: application/json' \
  -d '{"request_id":"demo-001","prompt":"請用一句話說明 resident inference。","max_new_tokens":64,"stream":false}'

curl -sS http://127.0.0.1:7860/api/v2/runs/demo-001
```

V2 request 使用 `max_new_tokens` 與 versioned contract；V1 的 `max_tokens`、`variants` 欄位會被拒絕，避免兩條協定混用。

### 案例 D：取消與 durable recovery

送出一個長請求後呼叫：

```bash
curl -sS -X POST http://127.0.0.1:7860/api/v2/runs/demo-001/cancel \
  -H 'Content-Type: application/json' -d '{}'
```

服務會將 cancellation 傳到 admission handle，等待 runner/cleanup 結束，再把 job 寫成 terminal cancelled。程序重啟時不會假裝續跑未完成 generation，而是依 recovery policy 收斂 durable records，讓 history 可被後續檢查。

### 相關文件

- [V2 serving ADR](docs/architecture/v2-serving-adr.md)
- [Admission / idempotency ADR](docs/architecture/v2-admission-adr.md)
- [Streaming ADR](docs/architecture/v2-streaming-adr.md)
- [LLMOps ADR](docs/architecture/v2-llmops-adr.md)
- [API integration ADR](docs/architecture/v2-integration-api-adr.md)
- [UI integration ADR](docs/architecture/v2-integration-ui-adr.md)
- [Runbook](INFERENCE_LAB.md)
- [Changelog](CHANGELOG.md)
