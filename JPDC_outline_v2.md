# JPDC Paper Outline — DASH (Async-KaaS) v2
## Target: Journal of Parallel and Distributed Computing (Regular Paper)
## Format: Elsevier single-column, ~18-20 pages excl. references
## Conference base: IEEE EDGE 2026 (KaaS-Edge, 6 pages)
## Required: ≥30% new material + Summary of Changes

---

## Working Title

**DASH: Deadline-Aware Straggler-Tolerant Scheduling for
Asynchronous Federated Distillation on Heterogeneous Edge Devices**

---

## Design Philosophy

**EDGE 版敘事軸**: "數學問題 → 解法 → 實驗驗證"
**JPDC 版敘事軸**: "分散式系統設計 → 每個模組解決什麼問題 → 數學服務於模組"

核心轉換：water-filling 和 submodularity 從主角降格為系統組件，
async protocol design、straggler handling、timeout policy 成為新主角。

---

## Section Structure

### Sec 1. Introduction (2 pages)

開頭從系統角度切入，不從 FD 通訊效率切入。

- **P1-2**: Edge computing 中的 straggler 問題
  - 異質性不只 data/model，更在 latency（CPU/channel/noise）
  - 同步 FL/FD：等最慢 device → round time 被 worst-case 支配
  - 量化：M=100 + 30% straggler → 同步 round time 膨脹 3-5x

- **P3-4**: 現有 async FL 為何不適用 FD
  - FedAsync/FedBuff 做 parameter aggregation，直接 weighted average
  - FD 傳的是 logits（soft labels），aggregation 語義不同
  - Partial logit upload 是 FD 獨有的問題：device 算完但只傳一半

- **P5-6**: KaaS 系統 + DASH 的設計理念
  - 不等最慢的，但也不浪費已收到的 partial knowledge
  - 排程階段就考慮 straggler probability（straggler-aware selection）
  - Adaptive timeout policy + quality-weighted aggregation
  - 系統層面容錯，不需要重新解最佳化

- **Contributions**（5 items）：
  1. Async FD protocol with three-outcome straggler model（complete / partial / timeout）
  2. **Straggler-aware device selection**：在 RADS greedy selection 中加入 completion probability，理論保證 $(1-p)(1-1/e)/2$ expected quality ratio
  3. Three timeout policies（fixed / adaptive / partial-accept）with system-level trade-off analysis
  4. Theoretical expected quality bound under straggler rate $p$（Theorem 2）
  5. 大規模實驗：2 datasets, M up to 200, straggler ratio 0-50%, wall-clock analysis

> **vs. 原版差異**：原版 contribution (2) 只有 timeout policy，沒有 straggler-aware scheduling。原版也沒有理論貢獻 (4)。這兩點是審稿人最可能質疑的地方。

---

### Sec 2. Related Work (2.5 pages)

- **2.1 Federated Distillation and Knowledge Transfer**
  - FedMD, FedDF, DS-FL, FedGEN, FedSKD...
  - 重寫敘述：強調「全部假設同步、full participation 或 random selection」
  - Gap: 沒有人在 FD 中處理 straggler / partial upload

- **2.2 Asynchronous Federated Learning** ← 全新
  - FedAsync (Xie et al., 2019)：async SGD + polynomial staleness weight
  - FedBuff (Nguyen et al., AISTATS 2022)：buffered async, K responses → one update
  - FedSA (Su & Li, 2023)：semi-async with deadline
  - AsyncFedED (Chen et al., 2023)：async with event-driven trigger
  - KAFL (Wu et al., 2023)：knowledge-aware async FL
  - Gap: **全部都是 parameter-level async**；logit-level async 無人探討

- **2.3 Client Selection and Straggler Mitigation**
  - Oort (Lai et al., OSDI 2021)：utility-based selection, 但不考慮 straggler probability
  - FedCS (Nishio & Yonetani, 2019)：deadline-aware selection
  - FedSRC, FedSCS (EDGE 引用的)
  - Coded computing / redundancy approaches
  - Gap: 沒有人把 straggler probability 整合進 submodular selection

---

### Sec 3. System Architecture (3 pages) ← 最大改動

**這一節讀起來像在介紹一個分散式系統設計，不是數學模型。**

- **3.1 System Overview** (1 page)
  - **Fig. 1**: 全新系統架構圖
    ```
    ┌──── Edge Server ────────────────────────────┐
    │  ┌─────────┐  ┌──────────────┐  ┌─────────┐ │
    │  │  RADS   │→│Async Controller│→│Aggregation│ │
    │  │Scheduler│  │(Timeout+Collect)│ │ Engine  │ │
    │  └─────────┘  └──────────────┘  └─────────┘ │
    └────────────────────┬────────────────────────┘
                         │ unreliable channel
           ┌─────────────┼─────────────┐
           ↓             ↓             ↓
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Device 1 │  │ Device 2 │  │ Device k │
    │ ✓ on-time│  │ ⚡partial │  │ ✗timeout │
    └──────────┘  └──────────┘  └──────────┘
    ```
  - Protocol 四階段：**Schedule → Dispatch → Collect (with deadline) → Aggregate**
  - 和同步版的差異：Phase 3 不等所有 device，Phase 4 用 quality-weighted aggregation

- **3.2 Device Cost and Quality Model** (0.5 page)
  - $b_i$: marginal per-sample upload cost
  - $q_i(v_i) = \rho_i \cdot v_i / (v_i + \theta_i)$: Michaelis-Menten quality with privacy degradation
  - $\rho_i \in (0, 1]$: privacy degradation factor (LDP noise effect)
  - 從 EDGE 搬過來，精簡呈現，只是「系統的參數介面」

- **3.3 Straggler Latency Model** (1 page) ← 全新
  - Device $i$ 的 round-trip latency：$\tau_i = \tau_i^{comp} + \tau_i^{comm} + \tau_i^{noise}$
    - $\tau_i^{comp} = c_i \cdot |D_i| \cdot E_{local}$（deterministic, device-dependent）
    - $\tau_i^{comm} = v_i \cdot s_{logit} / r_i$（deterministic, allocation-dependent）
    - $\tau_i^{noise} \sim \text{LogNormal}(\mu, \sigma)$（stochastic）
  - Deadline $D^{(t)}$ 由 ES 設定
  - 三種 outcome + 公式：
    - Complete: $\tau_i \leq D^{(t)} \Rightarrow v_i^{recv} = v_i^*$
    - Partial: $\tau_i^{comp} \leq D^{(t)} < \tau_i \Rightarrow v_i^{recv} = \lfloor v_i^* \cdot (D^{(t)} - \tau_i^{comp}) / \tau_i^{comm} \rfloor$
    - Timeout: $\tau_i^{comp} > D^{(t)} \Rightarrow v_i^{recv} = 0$
  - **Definition 1 (Straggler Rate)**：$p^{(t)} = |\{i \in S^{(t)} : \tau_i > D^{(t)}\}| / |S^{(t)}|$

- **3.4 Async Protocol** (0.5 page) ← 全新
  - **Algorithm 1**: DASH Async Protocol（pseudo-code）
  - 和同步版的 line-by-line 差異標示

---

### Sec 4. Scheduling and Aggregation (3 pages)

> **vs. 原版差異**：原版 Sec 4 只有 2.5 頁且 4.1 是「搬過來」。修改版把 4.2 擴充為 straggler-aware selection（新 Proposition），並加入 Theorem 2 作為理論貢獻。

- **4.1 Volume Allocation via Water-Filling** (0.75 page)
  - Proposition 1（closed-form water-filling）— 從 EDGE 搬過來
  - 敘述重寫：「RADS Scheduler 的第一階段使用以下 KKT-based 結果」

- **4.2 Straggler-Aware Device Selection** (1 page) ← 全新核心貢獻
  - EDGE 版：greedy maximize $Q(S) = \sum_{i \in S} \rho_i \cdot q_i(v_i^*)$
  - JPDC 版：greedy maximize **expected** quality：
    $$\tilde{Q}(S) = \sum_{i \in S} \rho_i \cdot q_i(v_i^*) \cdot \pi_i(D)$$
    其中 $\pi_i(D) = \Pr[\tau_i \leq D]$ 是 device $i$ 在 deadline $D$ 下的 completion probability
  - $\pi_i(D)$ 可從 straggler model 的 log-normal CDF 算出（closed-form）
  - **Proposition 2** (submodularity preserved)：$\tilde{Q}(S)$ 仍為 submodular + monotone，因為 $\pi_i(D)$ 是 per-device 常數（given D），乘上去不影響 submodularity
  - **Theorem 1** (approximation guarantee)：Greedy on $\tilde{Q}$ 仍有 $(1-1/e)/2$ ratio，但是是對 **expected** quality 而非 deterministic quality
  - **Theorem 2** (expected quality under straggler rate) ← 新理論貢獻：
    $$\mathbb{E}[\tilde{Q}(S^{greedy})] \geq (1-\bar{p}) \cdot \frac{1-1/e}{2} \cdot Q^*_{OPT}$$
    其中 $\bar{p}$ 為平均 straggler rate。
    直覺：completion probability 的平均值 $= 1 - \bar{p}$，乘上原本的 approximation ratio。
  - 討論：straggler-aware selection 自然偏好 reliable device，但 device diversity（submodular gain）會平衡這個偏好

- **4.3 Timeout Policy Design** (0.75 page) ← 全新
  - **Policy A — Fixed**: $D^{(t)} = D_0$ for all $t$
    - 優點：simple, predictable round time
    - 缺點：不適應 latency 變化
  - **Policy B — Adaptive**: $D^{(t)} = \text{percentile}_p(\{\tau_i^{(t-1)}\})$
    - $p$ 越高 → 等越久 → straggler 越少但 round 越慢
    - Warmup：前 $W$ 輪用 fixed deadline
  - **Policy C — Partial-Accept**: 和 B 相同的 adaptive deadline，但接受 partial logits
    - Partial device 的有效 quality：$q_i(v_i^{recv}) < q_i(v_i^*)$
    - 不浪費已完成的 computation

- **4.4 Quality-Weighted Aggregation** (0.5 page)
  - 收到 logits 後的 aggregation weight：
    $$w_i = \frac{\rho_i \cdot q_i(v_i^{recv})}{\sum_{j \in S^{recv}} \rho_j \cdot q_j(v_j^{recv})}$$
  - 注意：$v_i^{recv}$ 可能 $< v_i^*$（partial device），此時 $q_i$ 自然降低
  - 和 EDGE 的差異：EDGE 用 $v_i^*$，JPDC 用 $v_i^{recv}$
  - Distillation loss：和 EDGE 相同（KL divergence on public set）

> **關於 staleness**：本文的 async protocol 是 round-level async（每輪設 deadline），不是 pipeline async。所有收到的 logits 都是用當前輪的 server model 算的，因此 staleness = 0。這是系統設計的選擇，不是 limitation。
>
> **不建議硬塞 staleness parameter**：原版 spec 說 staleness = 0 但留一個 λ parameter，審稿人會覺得 half-baked。要嘛做一個完整的 multi-round staleness（太複雜），要嘛乾淨地不做。建議不做，在 Discussion 裡 explicitly acknowledge 並留給 future work。

---

### Sec 5. Performance Evaluation (5.5 pages) ← 主要承載區

- **5.1 Experimental Setup** (1 page)
  - **Datasets**:
    - CIFAR-100 (100 classes, 32×32, Dirichlet α=0.3)
    - EMNIST-ByClass (62 classes, 28×28→32×32, natural writer-based non-IID)
    > **vs. 原版差異**：原版用 FEMNIST（LEAF benchmark），但 LEAF 的原始格式很麻煩（JSON per user, 需另外下載處理）。改用 torchvision 內建的 EMNIST-ByClass，按 writer ID 分 partition，同樣是 naturally non-IID，但一行 `torchvision.datasets.EMNIST(split='byclass')` 就能載入，reviewability 和 reproducibility 都更好。
  - **Scale**: M ∈ {20, 50, 100, 200}
  - **Straggler model**: LogNormal noise, σ ∈ {0.0, 0.3, 0.5, 1.0, 1.5} → straggler rate ≈ 0-50%
  - **Baselines** (6 methods):
    | Method | Description |
    |---|---|
    | **DASH (ours)** | Straggler-aware RADS + adaptive timeout + D_min floor |
    | Sync-Greedy | 同步 RADS（π_i=1, 等所有人），DASH 的 ablation baseline |
    | FedBuff-FD | Buffered async adapted to FD：buffer K responses → one aggregation |
    | Random-Async | Random selection + fixed timeout |
    | Full-Async | All devices + adaptive timeout（no budget constraint）|
    | Sync-Full | All devices, synchronous（worst-case upper bound）|
  - 每組 3 seeds (42, 123, 456)
  - **Hardware**: RTX 5070 Ti, 16 GB VRAM

- **5.2 Main Comparison: Sync vs Async** (1.5 pages) — 對應 Experiment 1
  - **Fig. 2**: Accuracy vs Communication Round (CIFAR-100, M=50, σ=0.5)
    - 6 curves, one per method
    - DASH 和 Sync-Greedy per-round accuracy 幾乎重疊（兩者都用 quality-aware 機制）
    - Full-Async 有明顯 gap (~8pp lower)；FedBuff-FD 幾乎不學 (~17%)
  - **Fig. 3**: Accuracy vs Wall-Clock Time ← 這是 async 的殺手圖
    - 同樣 6 curves 但 x 軸是累積 wall-clock time
    - DASH 在 ~500s 達到 40%，Sync-Greedy 要 ~1,100s（3.14× speedup）
    - Sync-Greedy 的每個 round 在 x 軸上間距大（每輪等 ~54-68s）
  - **Table I**: Final accuracy, best accuracy, total wall-clock time, speedup
    - 6 methods × 4 metrics，DASH 44.47%/979s vs Sync-Greedy 43.66%/3,079s

- **5.3 Straggler Severity Sweep** (1 page) — 對應 Experiment 2
  - **Fig. 4**: Final accuracy vs straggler ratio σ ∈ {0, 0.3, 0.5, 1.0, 1.5}
    - DASH accuracy 只掉 1.2pp (44.68→43.47%)，對 straggler 非常 robust
    - Sync-Greedy accuracy 也穩定（因為等所有人）但 WC 膨脹
  - **Fig. 5**: Total wall-clock time vs straggler ratio
    - Sync-Greedy WC 隨 σ 膨脹 15.6%；DASH 只膨脹 13.4%
    - Speedup 從 3.15× (σ=0) → 3.20× (σ=1.5) — straggler 越嚴重，async 優勢越大

- **5.4 Timeout Policy Comparison** (0.75 page) — 對應 Experiment 3
  - **Fig. 6 左**: Accuracy vs wall-clock scatter (13 configs: 7 with D_min + 6 without)
    - adaptive(0.7)+D_min = Pareto 最優 (45.05%/871s)
    - fixed(5.0) 無 D_min → 崩潰到 35.96%（+D_min 救回 +7.58pp）
    - Partial-accept 反而掉精度 (-0.50pp)：不完整 logits 引入 noise
  - **Fig. 6 右**: Deadline evolution — 展示 spiral 現象 (adaptive 0.5 無 D_min 降到 5.6s)
  - Insight: **adaptive(p=0.7) + D_min floor 是 sweet spot**；D_min 是不可省略的安全機制

- **5.5 Scalability** (0.75 page) — 對應 Experiment 4
  - **Fig. 7**: Final Accuracy vs M (budget=2.5M, 50 rounds)
    - DASH 和 Sync-Greedy 都隨 M 增加但在 M=100 飽和 (~47.5%)
    - FedBuff-FD 完全不隨 M 變化 (~17%) — buffer K 固定
    - DASH 始終略高於 Sync-Greedy (+0.13 ~ +1.08pp)
  - **Fig. 8**: Total Wall-Clock Time vs M (+ Speedup curve)
    - Sync: WC 從 6,952s (M=20) → 10,079s (M=200)，成長 1.45×
    - DASH: WC 幾乎 flat，2,494s → 2,889s，僅 1.16×
    - Speedup: 2.79× (M=20) → 3.49× (M=200) — 單調遞增

- **5.6 Cross-Dataset Validation (EMNIST)** (0.5 page) — 對應 Experiment 5
  - **Fig. 9**: EMNIST accuracy vs wall-clock, M ∈ {50, 200}
    - 62 classes, naturally non-IID (writer-based partition)
    - 驗證結論在不同 dataset 上也成立（初步 seed42: DASH 81.49% > Sync-Greedy 77.47%）

- **5.7 Privacy Robustness under Async** (0.5 page) — 對應 Experiment 6
  - **Fig. 10**: Privacy ρ sweep under async conditions
    - 和 EDGE 的 Fig. 5 對比：async 條件下 privacy degradation pattern 類似

---

### Sec 6. Discussion (1.5 pages)

> **vs. 原版差異**：原版只列了 4 個 bullet points 共 1 頁。擴充為結構化的 discussion。

- **6.1 When to Use Sync vs Async**
  - 決策 guideline：if expected straggler rate > 15% and M > 30 → use DASH
  - 如果 devices 很 homogeneous（e.g., same hardware）→ Sync-Greedy 更簡單也夠好

- **6.2 Practical Deadline Tuning**
  - Adaptive policy 的 warmup 期選擇（W=3 rounds, safety=1.5）
  - Percentile p 的 sensitivity：p ∈ [0.7, 0.9] 都 robust（DASH 預設 p=0.85）
  - **D_min floor 的必要性**：Exp 3 證明無 D_min 的 fixed(5.0) 崩潰到 35.96% → 有 D_min 救回 +7.58pp

- **6.3 Relationship to Conference Version**
  - 明確聲明和 EDGE 版的差異
  - EDGE = KaaS-Edge, synchronous scheduling theory
  - JPDC = DASH, asynchronous system design with straggler tolerance

- **6.4 Limitations and Future Work**
  - 目前 straggler model 是 simulated（not real-world testbed）
  - Staleness 在本文為 0（round-level async, not pipeline async）
    → Future: pipeline async with multi-round staleness
  - Convergence rate bound under straggler → 需要 non-trivial 分析，explicitly 留給 future work
  - Extension to heterogeneous model architectures（不同 device 用不同 model）

---

### Sec 7. Conclusion (0.5 page)

---

### References (~30-35 entries)

新增引用（EDGE 沒有的）：
- Xie et al., "Asynchronous Federated Optimization" (2019)
- Nguyen et al., "Federated Learning with Buffered Asynchronous Aggregation" (AISTATS 2022)
- Su & Li, "Semi-Asynchronous FL with Adaptive Deadline" (2023)
- Lai et al., "Oort: Efficient Federated Learning via Guided Participant Selection" (OSDI 2021)
- Nishio & Yonetani, "Client Selection for FL with Heterogeneous Resources" (ICC 2019)
- Additional 5-8 async FL references

---

## 新舊內容比例估算

| Section | Pages | New % | Notes |
|---|---|---|---|
| Sec 1 Introduction | 2 | 80% | 角度完全不同 |
| Sec 2 Related Work | 2.5 | 50% | 2.2 全新 |
| Sec 3 System Architecture | 3 | 65% | 3.3, 3.4 全新 |
| Sec 4 Scheduling & Aggregation | 3 | 55% | 4.2 straggler-aware selection 全新, Thm 2 全新 |
| Sec 5 Experiments | 5.5 | 80% | 5.2-5.6 全新 |
| Sec 6-7 Discussion + Conclusion | 2 | 75% | 結構化 discussion |
| **Total** | **~18 pages** | **~65% new** |

---

## Summary of Changes (投稿時附上)

This manuscript extends our conference paper published at IEEE EDGE 2026 [ref]. The major extensions include:

1. **Asynchronous protocol design (DASH)** with a three-outcome straggler latency model (complete / partial / timeout), three timeout policies, and a $D_{\min}$ floor mechanism that prevents deadline spiral (Sec 3.3-3.4).

2. **Straggler-aware device selection** that integrates completion probability $\pi_i(D)$ into the submodular greedy selection via $\tilde{\rho}_i = \pi_i \cdot \rho_i$, with a new expected quality bound (Theorem 2) showing $(1-\bar{p})(1-1/e)/2$ approximation ratio under average straggler rate $\bar{p}$ (Sec 4.2).

3. **Comprehensive new experiments** (6 experiment groups, 3 seeds each):
   - Main comparison of 6 methods showing 3.14× speedup with no accuracy loss (Sec 5.2)
   - Straggler severity sweep σ ∈ {0–1.5} demonstrating robustness (Sec 5.3)
   - Timeout policy comparison with/without $D_{\min}$, identifying adaptive(0.7)+$D_{\min}$ as Pareto optimal (Sec 5.4)
   - Scalability to M=200 devices with speedup increasing from 2.79× to 3.49× (Sec 5.5)
   - Cross-dataset validation on EMNIST-ByClass (Sec 5.6)
   - Privacy robustness under async conditions (Sec 5.7)

4. **System-oriented restructuring**: the paper is reorganized from a distributed system design perspective, positioning the scheduling theory as a component within the overall async architecture.

Approximately 65% of the content is new.
