# JPDC Paper Outline — Async-KaaS-Edge
## Target: Journal of Parallel and Distributed Computing (Regular Paper)
## Format: Elsevier single-column, ~18-20 pages excl. references
## Conference base: IEEE EDGE 2026 (KaaS-Edge, 6 pages)
## Required: ≥30% new material + Summary of Changes

---

## Working Title

**Straggler-Tolerant Knowledge Distillation Scheduling for
Asynchronous Edge Intelligence Systems**

(Alternative: "Async-RADS: Asynchronous Resource-Aware Distillation
Scheduling with Straggler Tolerance for Distributed Edge Networks")

---

## Design Philosophy

**EDGE 版的敘事軸**: "我們有一個數學問題 → 解法 → 實驗驗證"
**JPDC 版的敘事軸**: "我們設計了一個分散式系統 → 系統的每個模組需要解決什麼問題 → 數學只是服務於模組設計"

核心轉換：把 water-filling 和 submodularity 從論文主角降格為系統的一個組件，
讓 async protocol design、straggler handling、timeout policy 成為新的主角。

---

## Section Structure

### Sec 1. Introduction (2 pages)
**角度完全不同於 EDGE 版。**

- 開頭：分散式邊緣系統的 straggler 問題（不是 FD 的通訊效率）
  - Edge computing 中的異質性不只是 data/model，還有 latency
  - 傳統同步 FL/FD 在等最慢的 device 時浪費大量時間
  - 現有 async FL 方法（FedAsync, FedBuff）針對 parameter aggregation，不適用於 logit-based distillation

- 中段：FD 在 async 環境下的獨特挑戰
  - Partial logit upload: device 只傳了一部分就斷線/超時
  - Staleness: 用舊 model 產生的 logits 混進當前 round
  - Quality degradation: straggler 的 logit 品質可能比新 device 差

- KaaS-Edge 系統 + Async-RADS 的設計理念：
  - 不等最慢的，但也不浪費已收到的 partial knowledge
  - Timeout-aware scheduling + staleness-weighted aggregation
  - 系統層面的容錯，不需要額外的數學最佳化理論

- 貢獻列表（4 items）：
  1. 設計 async FD protocol with straggler tolerance
  2. 提出 timeout policy（fixed/adaptive/partial-accept）
  3. 擴展 RADS with staleness-aware quality weighting
  4. 大規模實驗（2 datasets, M up to 200, 多種 straggler 比例）

### Sec 2. Related Work (2.5 pages)
**重組為三個子 thread：**

- 2.1 Federated Distillation
  - 保留 EDGE 的引用但重寫敘述，強調 "都假設同步"
  
- 2.2 Asynchronous Federated Learning
  - **全新段落**。引用：
    - FedAsync (Xie et al., 2019) — async SGD + staleness weight
    - FedBuff (Nguyen et al., AISTATS 2022) — buffered async aggregation
    - FedSA (Su & Li, 2023) — semi-async with deadline
    - AsyncFLEO (Elmahallawy & Luo, 2022) — async FL in satellite (和 EDGE 裡的 FedSCS 論文呼應)
  - Gap: 這些都是 parameter-level async，沒有人做 logit-level async

- 2.3 Client Selection and Resource Scheduling
  - 保留 EDGE 的 Balakrishnan、Zhang TOSN、FedSRC、FedSCS
  - 加入 straggler-aware scheduling 文獻

### Sec 3. System Architecture (3 pages) ← **這是最大的改動**
**不叫 "System Model"，叫 "System Architecture"，像在介紹一個系統設計。**

- 3.1 System Overview (1 page)
  - 全新的系統架構圖（Fig. 1）：
    - ES 側：RADS Module + Async Controller + Aggregation Engine
    - Device 側：Local Trainer + Logit Generator + Upload Manager
    - 中間：Unreliable Channel（標示 latency distribution, dropout）
  - Protocol 的 4 個 phase：Schedule → Dispatch → Collect (with timeout) → Aggregate
  - 明確畫出 straggler 的三種結果：on-time / partial / timeout

- 3.2 Cost and Quality Model (0.5 page)
  - b_i, a_i, q_i(v_i) — 從 EDGE 搬過來但精簡
  - 這裡只是「系統的參數介面」，不是論文的重點

- 3.3 Straggler Model (1 page) ← **全新**
  - Device i 的 round-trip latency: τ_i = τ_comp + τ_comm + τ_noise
    - τ_comp: 取決於 local dataset 大小 + CPU frequency
    - τ_comm: 取決於 v_i × payload size / channel rate
    - τ_noise: 隨機擾動（log-normal distribution）
  - ES 設定 per-round deadline D^(t)
  - 三種 outcome：
    - Complete: τ_i ≤ D^(t)，收到 v_i* 完整 logits
    - Partial:  τ_comp ≤ D^(t) < τ_i，收到 v_i^recv < v_i* 的 logits
    - Timeout:  τ_comp > D^(t)，什麼都沒收到

- 3.4 Async Protocol Design (0.5 page) ← **全新**
  - Algorithm: Async-KaaS Protocol（新的 Algorithm 2）
  - 和 EDGE 版的 Algorithm 2 差異：加入 timeout 判斷、partial collect、staleness check

### Sec 4. Scheduling Algorithms (2.5 pages)
**把 EDGE 的 Sec IV "RADS" 降格為這裡的一個子模組。**

- 4.1 Base RADS: Water-Filling + Greedy (1 page)
  - Proposition 1 (water-filling) — 搬過來
  - Proposition 2 (submodularity) — 搬過來
  - Theorem 1 (approximation) — 搬過來
  - 但**敘述方式改變**：不是「我們推導了一個定理」，而是
    「RADS Module 的排程核心使用了以下已建立的結果」

- 4.2 Staleness-Aware Quality Weighting (0.75 page) ← **全新**
  - 當 device i 回傳的 logits 是用 model w^(t-s) 算的（s = staleness）：
    - Adjusted quality: q̃_i = q_i(v_i^recv) / (1 + λ·s)
    - λ 是 staleness discount factor
  - Aggregation weight: w_i ∝ ρ_i · q̃_i
  - 系統邏輯，不需要新定理

- 4.3 Timeout Policy Design (0.75 page) ← **全新**
  - Policy A — Fixed Deadline: D^(t) = D_0 for all t
  - Policy B — Adaptive Deadline: D^(t) = percentile_p(τ^(t-1))
    - p-th percentile of previous round's latency distribution
    - p 越高 → 等越久 → straggler 越少但 round 越慢
  - Policy C — Partial-Accept: D^(t) adaptive + 接受 partial logits
  - 三種 policy 的 trade-off 用實驗比較，不用數學分析

### Sec 5. Performance Evaluation (5-6 pages)
**大幅擴展，是新貢獻的主要承載區。**

- 5.1 Experimental Setup (1 page)
  - Datasets: CIFAR-100 + FEMNIST
  - Scale: M ∈ {20, 50, 100, 200}
  - Straggler model: log-normal latency, straggler ratio ∈ {0%, 10%, 30%, 50%}
  - Baselines:
    - Sync-RADS (EDGE 版, 等所有人)
    - FedBuff-FD (buffered async, adapted to FD)
    - Random-Async (random selection + fixed timeout)
    - Full-Participation-Async (all devices, with timeout)
  - 每組 3 seeds

- 5.2 Convergence under Straggler Conditions (1.5 pages) ← **全新**
  - Fig: Accuracy vs Round at straggler ratio = 0%, 10%, 30%, 50%
  - Fig: Accuracy vs Wall-clock Time（不是 round，是真實時間！）
  - Key insight: Sync-RADS 的 per-round accuracy 最高但 wall-clock 最慢
  - Async-RADS at 30% straggler 在 wall-clock 上比 Sync 快 2-3x

- 5.3 Timeout Policy Comparison (1 page) ← **全新**
  - Fig: 三種 policy 的 accuracy vs time trade-off
  - Fig: Partial-accept 的 received logit fraction vs accuracy
  - Table: Fixed vs Adaptive vs Partial-Accept 的 summary

- 5.4 Scalability (1 page) ← **全新**
  - Fig: M=20, 50, 100, 200 的 convergence curves
  - Fig: Per-round scheduling time vs M（展示 O(M²log(1/δ)) 的實際數字）
  - Key insight: 同步在 M=200 時幾乎不可用，async 仍穩定

- 5.5 Communication Efficiency (0.5 page)
  - 搬 EDGE 的 Fig 3 (cumulative comm) 但加 async 的曲線
  - Async-RADS 因為跳過 straggler 所以 cumulative comm 更低

- 5.6 Privacy Robustness (0.5 page)
  - 搬 EDGE 的 Fig 5 (ρ sweep) 但在 async 條件下重跑

### Sec 6. Discussion (1 page)
- Sync vs Async 的適用場景分析
- Deadline D^(t) 的 practical tuning guideline
- Limitation: 沒有 convergence bound（explicitly stated as future work with TMC-style theory）
- 和 EDGE conference 版的差異聲明

### Sec 7. Conclusion (0.5 page)

### References (~30-35 entries)

---

## 新舊內容比例估算

| 內容 | 頁數 | 來源 |
|---|---|---|
| Sec 1 Introduction | 2 | 80% 全新 |
| Sec 2 Related Work | 2.5 | 50% 新（2.2 全新） |
| Sec 3 System Architecture | 3 | 60% 新（3.3, 3.4 全新） |
| Sec 4 Scheduling | 2.5 | 40% 新（4.2, 4.3 全新） |
| Sec 5 Experiments | 5.5 | 75% 新（5.2-5.4 全新） |
| Sec 6-7 Discussion + Conclusion | 1.5 | 70% 新 |
| **Total** | **~17 pages** | **~60% 新** |

遠超 JPDC 的 30% 最低要求。

---

## Summary of Changes (投稿時附上)

This manuscript extends our conference paper published at
IEEE EDGE 2026 [ref]. The major extensions include:

1. A new asynchronous protocol design with straggler tolerance
   (Sec 3.3-3.4), including straggler latency modeling and
   three timeout policies (Sec 4.3).

2. Staleness-aware quality weighting for aggregating partial
   and delayed logit contributions (Sec 4.2).

3. Comprehensive new experiments: a second dataset (FEMNIST),
   scale-up to M=200 devices, straggler ratio sweep,
   wall-clock time analysis, and timeout policy comparison
   (Sec 5.2-5.4).

4. Complete restructuring of the paper from a system design
   perspective, repositioning the mathematical results as
   components within the system architecture.

Approximately 60% of the content is new.
