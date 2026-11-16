# V2 最小可打通 E2E Spec

**狀態：** Proposed

**目的：** 在單機單 GPU 的地端環境，打通一條可重跑、可追溯、可驗收的推論與評測流程，作為後續 Qwen3.8-27B 部署與優化的固定起點。

**適用版本：** `v2.0.0` 以上

## 1. 目標與邊界

### 目標

完成以下最小閉環：

```text
固定模型與 runtime identity
  → 啟動 V2 resident inference service
  → 執行一個真實 GPU 推論
  → 保存 request、response、resource 與 provenance evidence
  → 由 evaluation pipeline 讀取 evidence
  → 產生 baseline report
  → 通過 LLMOps evaluation gate
```

此閉環完成後，才開始調整 GPU layers、context size、batch 或 thread 等優化參數。每次優化都必須能回到同一份 baseline 做比較。

### 範圍內

- 單機、單張 GPU、loopback HTTP。
- Qwen3.8-27B GGUF 與目前 pinned llama.cpp runtime。
- V2 resident service 的啟動、非串流請求、串流請求、取消、deadline 與 durable job。
- 固定小型 evaluation dataset、推論效能指標、GPU/RAM 取樣與 evidence digest。
- 本地 registry、離線 evaluation gate 與可重跑 runbook。

### 範圍外

- Kubernetes、autoscaling、multi-GPU scheduling、distributed queue。
- LAN/public exposure、TLS、多人身份驗證。
- 動態 batching、模型自動下載、多模態生成。
- 自動 promotion 到 production 或遠端 model registry。

## 2. 兩條 pipeline 的責任

### Inference pipeline

```text
Client
  → web.server
  → V2 contracts
  → AdmissionQueue
  → ResidentSupervisor
  → llama-server
  → response / StreamEvent
  → DurableJobStore + Observability
```

Inference pipeline 只負責正確執行請求並產生可追溯結果。它不決定模型品質是否通過，也不負責 baseline comparison。

### Evaluation pipeline

```text
Versioned dataset + experiment config
  → Evaluation runner
  → Inference adapter (cold 或 resident)
  → immutable run evidence
  → quality / latency / throughput / memory metrics
  → baseline comparison
  → LLMOps evaluation gate
  → report + registry lineage
```

Evaluation pipeline 可以呼叫 inference pipeline，但兩者使用 versioned DTO、`run_id`、`request_id` 與 evidence reference 交接。Serving implementation 不應 import evaluation scoring code。

目前 repository 已有 V1 baseline/comparison 與 V2 LLMOps gate；尚缺一個統一的 evaluation adapter，讓同一份 dataset 能以 V1 cold 與 V2 resident 兩種模式執行。這是本 spec 的第一個實作缺口。

## 3. 最小 E2E 流程

1. **Freeze identity**：確認 GPU、driver/CUDA、llama.cpp revision、GGUF size/SHA-256、config SHA-256、dataset SHA-256。
2. **Preflight**：執行本機 readiness checks；任何 identity mismatch 直接停止。
3. **Start resident service**：以固定 config 啟動 `--execution-mode resident`，state path 與 evidence root 必須明確指定。
4. **Inference smoke**：依序驗證 `/api/v2/info`、非串流 request、stream request、cancel、deadline，以及 terminal state 在 cleanup 後才發布。
5. **Collect evidence**：每個 run 保存 request metadata、response/events、latency、tokens、GPU peak、RAM peak、runtime/model/config identity、source commit 與 digest。
6. **Evaluate**：固定 dataset 由 evaluation adapter 呼叫 V1 cold 或 V2 resident，產生逐案例結果與 aggregate report。
7. **Gate**：registry 驗證 manifest lineage；evaluation gate 檢查 quality、latency ratio、GPU peak、evidence digest 與 provenance。
8. **Publish baseline**：只有 gate pass 的 run 才可標成 baseline；後續優化以 `parent_run_id` 指向此 baseline。

## 4. Ticket 拆分

Ticket 依依賴順序執行。每張 ticket 都需要 focused acceptance，最後才執行一次完整 regression。

### E2E-01：固定地端 identity 與 deployment manifest

**目標：** 建立實際 GPU 的不可變環境基線。

**輸入：** GPU、driver/CUDA、pinned llama.cpp、GGUF、兩份現有 config、`experiments/tasks-v1.json`。

**交付：** versioned deployment manifest、runbook 欄位、model/runtime/config/dataset SHA-256。

**驗收：**

- A：manifest 缺少任一 identity 或 hash 時拒絕執行。
- A：模型檔或 runtime revision 不匹配時 preflight fail-closed。
- B：硬體資訊可讀但無法取得完整溫度或功耗欄位時，記錄 unavailable，不阻擋基本 smoke。

**依賴：** 無。

### E2E-02：真實 GPU V2 resident smoke

**目標：** 在目標地端 GPU 打通 V2 server lifecycle。

**交付：** 一次可重跑的 resident 啟動命令與 smoke receipt。

**驗收：**

- A：`/api/v2/info` 顯示 ready、固定 model/config identity。
- A：非串流請求回傳正確 `request_id`、`run_id`、contract version 與 terminal response。
- A：串流請求產生有序 `StreamEvent` 並且只有一個 terminal event。
- A：cancel/deadline 後，owned llama-server process group 已停止，cleanup 確認後才發布 terminal state。
- A：V1 欄位混入 V2 request 被拒絕，不能 fallback 到 `/api/*`。
- B：首次啟動耗時與 filesystem cache 狀態先記錄，暫不作硬性性能門檻。

**依賴：** E2E-01。

### E2E-03：Evaluation adapter 與統一 run schema

**目標：** 讓同一 dataset 可選擇 V1 cold 或 V2 resident，並產生相同 evidence schema。

**交付：** `InferenceAdapter` interface、cold adapter、resident adapter、run manifest/receipt schema、adapter contract tests。

**驗收：**

- A：同一 case ID、prompt hash、model/config/dataset identity 在兩種模式都能追溯。
- A：adapter failure、truncated output、missing metrics 不能被標成成功。
- A：每個 run 有唯一 `run_id`，重試不會產生錯誤的 duplicate terminal result。
- B：第一版只支援文字、非工具呼叫與固定 dataset；未支援能力明確標記 unavailable。

**依賴：** E2E-01、E2E-02。

### E2E-04：真實 GPU baseline matrix

**目標：** 取得可比較的 V1/V2 baseline，而不是直接優化。

**交付：** 至少三組 run：V1 24 layers、V1 40 layers、V2 resident；每組使用相同 dataset、generation settings 與硬體。

**驗收：**

- A：每組 run 都有完整 identity、evidence digest、quality status 與 resource samples。
- A：固定 dataset 的 case cardinality、重複次數與設定一致；缺 case 或重複 case fail-closed。
- A：report 清楚區分 cold startup、warm request latency、tokens/sec、VRAM peak、RAM peak。
- B：尚未有統計顯著性；第一版只報告固定重複次數與限制。

**依賴：** E2E-03。

### E2E-05：LLMOps evaluation gate 接線

**目標：** 將 baseline/comparison 結果接到 registry 與 gate，形成可判定的 pass/fail。

**交付：** gate input mapping、threshold config、registry lineage、gate report 與 CI/offline replay command。

**驗收：**

- A：missing/tampered evidence、identity mismatch、unverified provenance、quality/latency/GPU threshold failure 都 fail-closed。
- A：只有 gate pass 的 run 能標記為 accepted baseline 或 optimization candidate。
- A：candidate 必須以 `parent_run_id` 指向已接受 baseline。
- B：registry 先使用本地 append-only JSONL；多人並行寫入與遠端 dashboard 延後。

**依賴：** E2E-04。

### E2E-06：地端部署與優化 runbook

**目標：** 讓後續優化可以按照同一流程重做。

**交付：** deployment checklist、baseline command、optimization matrix、rollback steps、known limitations。

**驗收：**

- A：新環境能依 runbook 重建 identity、啟動 service、執行 smoke 與取得 baseline receipt。
- A：每次優化只改一個 declared factor，且保留 parent baseline 與 comparison report。
- B：目前只支援 loopback；LAN exposure、TLS、auth 另開 ticket。

**依賴：** E2E-01 至 E2E-05。

## 5. 階段 gate 與分工

| Gate | 出口條件 | 分級 | 驗收者 |
| --- | --- | --- | --- |
| G0 identity | manifest、hash、preflight 全部通過 | A | 獨立驗收者 |
| G1 inference | V2 real GPU smoke 全部 lifecycle checks 通過 | A | 獨立驗收者 |
| G2 evidence | run schema、digest、resource samples 完整 | A | 獨立驗收者 |
| G3 evaluation | baseline matrix 可重跑、比較欄位一致 | A | 獨立驗收者 |
| G4 LLMOps | registry lineage 與 gate pass report 產生 | A | 獨立驗收者 |
| G5 deployment | runbook 可由另一人重建 smoke/baseline | A | 獨立驗收者 |
| B backlog | 統計顯著性、遠端 registry、LAN auth、dashboard | B | 協調員記錄期限 |

規劃者負責 ticket scope 與 gate；執行者負責實作與 receipts；驗收者不得是同一張 ticket 的執行者。真實 GPU run 的次數、時間與成本先在 ticket 中固定；超過預算需重新裁決。

## 6. 完成定義

當 E2E-01 至 E2E-06 全部通過，專案才可宣稱「地端 inference/evaluation 最小閉環已打通」。在此之前，可以宣稱 V2 foundation 與 offline contracts 通過，但不能宣稱目標 GPU 上的完整服務與效能 baseline 已完成。
