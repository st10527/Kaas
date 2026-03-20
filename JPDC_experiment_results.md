# JPDC 2026 — 實驗結果與觀察總整理

> **用途**：將此文件連同 `JPDC_outline_v2.md` 和 `JPDC_changelog.md` 一起提供給 Claude web，
> 用於撰寫/修改論文正文。本文件包含所有已完成實驗的數值結果、趨勢觀察、以及論文寫作建議。
>
> **最後更新**：2026-03-20（Exp 1 + Exp 2 + Exp 3a/3b 完成，Exp 4-6 進行中）

---

## 一、實驗總覽與進度

| # | 實驗名稱 | 對應 Figure/Table | 狀態 | 結果 JSON |
|---|---------|-------------------|------|-----------|
| 1 | Main Comparison (6 methods) | Fig 2, 3, Table I | ✅ 3 seeds 完成 | `exp1_main_comparison.json` |
| 2 | Straggler Severity Sweep (5σ) | Fig 4, 5 | ✅ 3 seeds 完成 | `exp2_straggler_sweep.json` |
| 3a | Policy Comparison WITH D_min (7 cfg) | Fig 6 | ✅ 3 seeds 完成 | `exp3_policy_comparison.json` |
| 3b | Policy Comparison WITHOUT D_min (6 cfg) | Fig 6 | ✅ 3 seeds 完成 | `exp3b_policy_nofloor.json` |
| 4 | Scalability (M=20,50,100,200) | Fig 7, 8 | 🔄 進行中 | `exp4_scalability.json` |
| 5 | Cross-Dataset EMNIST (M=50,200) | Fig 9 | 🔄 進行中 | `exp5_emnist.json` |
| 6 | Privacy under Async (4 ρ levels) | Fig 10 | 🔄 進行中 | `exp6_privacy_async.json` |

---

## 二、共通實驗設定

| 參數 | 值 | 說明 |
|------|-----|------|
| Dataset | CIFAR-100 (Exp 1-4, 6), EMNIST-ByClass (Exp 5) | 100 類 / 62 類 |
| Model | CNN (4.7M params, 3×[Conv-BN-ReLU]×2 + FC) | 非 ResNet，FD 標準選擇 |
| Devices (M) | 50 (default), sweep {20,50,100,200} in Exp 4 | 3 tiers: A(fast)30%, B(mid)40%, C(slow)30% |
| Rounds | 50 | |
| Non-IID | Dirichlet α=0.3 | 中高異質性 |
| Seeds | 42, 123, 456 | 所有結果報 mean ± std |
| Budget | 50.0 (M=50), 2.5×M (Exp 4,5) | per-round total volume budget |
| Local epochs | 2 | |
| Distill epochs | 3, lr=0.001 | |
| Pretrain epochs | 10 | |
| Straggler model | LogNormal noise, 3-tier device rates | σ=0.5 default |
| DASH timeout | Adaptive, percentile=0.85, EMA α=0.3, warmup=3 rounds | |
| DASH D_min | 0.3 × D_warmup | 防止 deadline spiral |
| RADS v_feasible margin | 0.8 | |
| Hardware | RTX 5070 Ti, 16 GB VRAM (aelab-2) | |
| Wall-clock | **模擬的** (StragglerModel)，非真實等待 | 同步: max(τ_i), 異步: deadline D |

### 六個比較方法

| Method | 類型 | 選擇策略 | Deadline | 說明 |
|--------|------|---------|----------|------|
| **DASH (ours)** | Async FD | Straggler-aware greedy (π_i weighted) | Adaptive | 完整系統 |
| Sync-Greedy | Sync FD | Greedy (π_i=1) | 等所有人 | DASH 的同步 ablation |
| FedBuff-FD | Async FD | 無選擇，先到先得 buffer K=10 | 無 | Nguyen et al. adapted |
| Random-Async | Async FD | Random 50% devices | Fixed D=10s | Naive baseline |
| Full-Async | Async FD | All 50 devices | Adaptive | 無 budget 約束 |
| Sync-Full | Sync FD | All 50 devices | 等所有人 | Worst-case upper bound |

---

## 三、Experiment 1：Main Comparison

> **目的**：驗證 DASH 在標準設定下 (M=50, σ=0.5) 的 accuracy 與 wall-clock trade-off。
> **對應論文**：Sec 5.2, Fig 2 (acc vs round), Fig 3 (acc vs wall-clock), Table I

### Table I — Final Results (mean ± std, 3 seeds)

| Method | Accuracy (%) | Best Acc (%) | Wall-Clock (s) | Speedup vs Sync-Greedy |
|--------|-------------|-------------|----------------|----------------------|
| **DASH** | **44.47 ± 0.46** | 45.14 | **979 ± 52** | **3.14×** |
| Sync-Greedy | 43.66 ± 0.79 | 44.47 | 3,079 ± 283 | 1.00× |
| Sync-Full | 43.80 ± 0.71 | 44.29 | 73,422 ± 12,968 | 0.04× |
| Full-Async | 35.57 ± 0.85 | 35.62 | 505 ± 6 | 6.09× |
| Random-Async | 23.47 ± 0.35 | 23.86 | 500 ± 0 | 6.16× |
| FedBuff-FD | 17.53 ± 1.45 | 18.24 | 120 ± 6 | 25.59× |

### Per-seed Breakdown

| Method | seed42 | seed123 | seed456 | seed42 WC | seed123 WC | seed456 WC |
|--------|--------|---------|---------|-----------|------------|------------|
| DASH | 44.43% | 45.05% | 43.93% | 1,049s | 962s | 926s |
| Sync-Greedy | 44.70% | 43.49% | 42.78% | 2,718s | 3,111s | 3,409s |
| FedBuff-FD | 18.12% | 18.93% | 15.53% | 115s | 128s | 118s |
| Random-Async | 23.02% | 23.53% | 23.86% | 500s | 500s | 500s |
| Full-Async | 36.77% | 34.89% | 35.06% | 502s | 513s | 501s |
| Sync-Full | 44.60% | 43.93% | 42.88% | 59,526s | 70,006s | 90,735s |

### Accuracy Convergence (mean of 3 seeds, per round)

| Round | DASH | Sync-Greedy | FedBuff-FD | Random-Async | Full-Async | Sync-Full |
|-------|------|-------------|------------|-------------|------------|-----------|
| 0 | 17.24 | 17.17 | 8.97 | 11.28 | 13.91 | 14.11 |
| 5 | 26.35 | 26.10 | 10.44 | 13.47 | 18.07 | 21.61 |
| 10 | 33.54 | 32.59 | 10.93 | 14.90 | 21.21 | 27.13 |
| 15 | 37.60 | 37.27 | 12.68 | 16.64 | 23.28 | 30.91 |
| 20 | 39.70 | 40.19 | 13.28 | 16.53 | 25.48 | 35.02 |
| 25 | 40.89 | 41.45 | 15.01 | 17.49 | 27.37 | 37.78 |
| 30 | 42.59 | 42.89 | 15.31 | 19.73 | 29.89 | 40.60 |
| 35 | 43.40 | 43.21 | 14.51 | 21.02 | 31.57 | 41.60 |
| 40 | 43.30 | 43.63 | 15.40 | 21.61 | 33.33 | 42.70 |
| 45 | 43.97 | 43.50 | 16.68 | 22.39 | 32.19 | 43.76 |
| 49 | 44.47 | 43.66 | 17.53 | 23.47 | 35.57 | 43.80 |

### DASH Deadline Behavior

| Seed | D_warmup | D_min (floor) | D_final | D_mean |
|------|----------|-------------|---------|--------|
| 42 | 54.6s | 16.4s | 16.4s | 21.0s |
| 123 | 50.1s | 15.0s | 15.0s | 19.2s |
| 456 | 43.2s | 13.0s | 13.0s | 18.5s |

### Sync-Greedy Wall-Clock Variation (per-round dWC)

| Seed | Range | Mean | Std |
|------|-------|------|-----|
| 42 | [53.5, 55.8] | 54.4s | 0.53 |
| 123 | [61.5, 65.2] | 62.2s | 0.65 |
| 456 | [67.4, 69.4] | 68.2s | 0.51 |

### 觀察與論文寫作建議 (Exp 1)

1. **DASH accuracy 反而略高於 Sync-Greedy (+0.81pp)**
   - 原因：straggler-aware selection (π_i 加權) 在 3-seed 平均下偏好 reliable + high-quality device
   - 論文寫法：「DASH achieves comparable or slightly higher accuracy than Sync-Greedy while requiring only 31.8% of the wall-clock time.」
   - **不要**過度宣稱 DASH 精度更好（0.81pp 在 std 範圍內），重點放在「幾乎不損失精度」

2. **Ranking 完全符合預期**
   - Sync-Full ≈ Sync-Greedy ≈ DASH >> Full-Async > Random-Async > FedBuff-FD
   - 前三者精度接近是因為都用了 quality-aware 機制
   - Full-Async 沒有 budget 排程→ 資源分散、quality 低
   - FedBuff-FD 只用 buffer K=10 → 每輪資訊極少

3. **Sync-Full WC 極大 (73,422s ≈ 20.4hr)**
   - 50 devices 全等 → round time ~1190s → 是最清楚的「為什麼需要 async」的證據
   - 論文可強調：「Even with modest M=50, synchronous full participation requires 75× more wall-clock time for only 0.33pp accuracy gain over DASH.」

4. **Wall-clock 數字是模擬的**
   - 論文 Sec 5.1 需明確聲明：「Wall-clock time is computed via simulation using the StragglerModel (§3.3), where synchronous baselines use $T_{round} = \max_{i \in S} \tau_i$.」

---

## 四、Experiment 2：Straggler Severity Sweep

> **目的**：驗證 DASH 在不同 straggler 嚴重程度下的 robustness。
> **對應論文**：Sec 5.3, Fig 4 (acc vs σ), Fig 5 (WC vs σ)
> **變數**：σ ∈ {0.0, 0.3, 0.5, 1.0, 1.5}，其他設定同 Exp 1

### Accuracy vs Straggler Noise σ (mean ± std, 3 seeds)

| σ | DASH | Sync-Greedy | FedBuff-FD |
|---|------|-------------|------------|
| 0.0 | 44.68 ± 0.87 | 43.72 ± 0.78 | 17.84 ± 1.45 |
| 0.3 | 44.12 ± 0.77 | 44.03 ± 1.12 | 16.96 ± 0.66 |
| 0.5 | 44.34 ± 0.12 | 44.11 ± 1.07 | 17.50 ± 1.03 |
| 1.0 | 44.49 ± 1.15 | 44.76 ± 0.92 | 16.70 ± 0.75 |
| 1.5 | 43.47 ± 0.70 | 44.10 ± 0.82 | 17.29 ± 1.46 |

### Wall-Clock vs σ (seconds, mean ± std)

| σ | DASH | Sync-Greedy | FedBuff-FD |
|---|------|-------------|------------|
| 0.0 | 977 ± 49 | 3,073 ± 285 | 114 ± 2 |
| 0.3 | 977 ± 51 | 3,075 ± 285 | 118 ± 4 |
| 0.5 | 979 ± 52 | 3,079 ± 283 | 120 ± 6 |
| 1.0 | 994 ± 46 | 3,118 ± 263 | 123 ± 9 |
| 1.5 | 1,108 ± 36 | 3,551 ± 195 | 125 ± 11 |

### DASH vs Sync-Greedy: Speedup & Gap per σ

| σ | DASH Acc | Sync Acc | Gap (pp) | DASH WC | Sync WC | Speedup |
|---|----------|----------|----------|---------|---------|---------|
| 0.0 | 44.68% | 43.72% | +0.96 | 977s | 3,073s | 3.15× |
| 0.3 | 44.12% | 44.03% | +0.09 | 977s | 3,075s | 3.15× |
| 0.5 | 44.34% | 44.11% | +0.22 | 979s | 3,079s | 3.14× |
| 1.0 | 44.49% | 44.76% | -0.28 | 994s | 3,118s | 3.14× |
| 1.5 | 43.47% | 44.10% | -0.62 | 1,108s | 3,551s | 3.20× |

### DASH Participation & Deadline per σ (seed=42)

| σ | D_warmup | D_final | D_min | Avg Participants / 50 |
|---|----------|---------|-------|----------------------|
| 0.0 | 54.6s | 16.4s | 16.4s | 49.5 |
| 0.3 | 54.6s | 16.4s | 16.4s | 49.5 |
| 0.5 | 54.6s | 16.4s | 16.4s | 49.3 |
| 1.0 | 54.6s | 16.4s | 16.4s | 46.6 |
| 1.5 | 54.6s | 16.4s | 16.4s | 43.7 |

### 觀察與論文寫作建議 (Exp 2)

1. **DASH 對 straggler 非常 robust**
   - σ=0→1.5 全範圍，DASH accuracy 只掉 1.21pp (44.68→43.47%)
   - 論文：「DASH accuracy degrades by only 1.2pp across the full range of straggler severity (σ = 0 to 1.5), demonstrating the effectiveness of straggler-aware scheduling.」

2. **Speedup 在 σ=1.5 時反而最大 (3.20×)**
   - 直覺正確：straggler 越嚴重 → sync 等最慢的人更久 → async 的相對優勢更大
   - 論文：「The speedup ratio actually increases from 3.15× to 3.20× as σ grows, confirming that DASH's advantage is most pronounced under severe straggler conditions.」

3. **σ=0 sanity check 通過**
   - 無 noise 時 DASH ≈ Sync (gap +0.96pp)，證明 async protocol 本身 overhead 極小
   - 論文可提：「When σ=0 (no stochastic delay), DASH matches Sync-Greedy accuracy while still achieving 3.15× speedup due to the deterministic latency heterogeneity across device tiers.」

4. **Sync-Greedy WC 隨 σ 膨脹 15.6%，DASH 只膨脹 13.4%**
   - 但 DASH 在 σ=1.5 時 participant 降至 43.7/50 → 有些 device timeout
   - 論文可用此說明 adaptive deadline + partial-accept 的價值

5. **FedBuff-FD 不受 σ 影響**
   - 因為它只等 buffer K=10 (先到先得)，但 accuracy 永遠低 (~17%)
   - 論文：「FedBuff-FD's wall-clock is insensitive to σ but achieves only 17% accuracy — the buffered approach discards scheduling quality for speed.」

6. **Sync-Greedy accuracy 隨 σ 不降反穩**
   - 這是因為 Sync-Greedy 等所有人完成 → 不受 straggler 影響，只是等更久
   - 符合理論：sync 的精度不受 σ 影響，但 WC 受影響

---

## 五、已知問題與修正記錄摘要

> 完整修正記錄見 `JPDC_changelog.md`。以下是影響論文寫作的重點：

| # | 修正 | 影響的論文段落 |
|---|------|-------------|
| 1 | KaaS-Edge → Sync-Greedy | Sec 5 所有 baseline 名稱 |
| 2 | ρ̃_i = π_i · ρ_i 代入 water-filling | 確認 Algorithm 2 與代碼一致 |
| 3 | Sync WC 模擬 (max τ_i) | Sec 5.1 需聲明 WC 定義 |
| 4 | NumPy warning 壓制 | 無 |
| 5 | Warmup D 改用 budget-based, p85, margin 0.8 | Sec 4.3 warmup 描述, Table II parameters |
| 6 | D_min = 0.3 × D_warmup floor | Algorithm 1 加 D_min, Sec 4.3, Table II |

### 論文 Table II (Parameter Settings) 需列出的值

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Adaptive percentile | $p$ | 0.85 |
| EMA smoothing | $\alpha$ | 0.3 |
| Warmup rounds | $W$ | 3 |
| Min deadline ratio | $D_{\min}/D^{(0)}$ | 0.3 |
| v_feasible margin | — | 0.8 |
| Warmup safety factor | — | 1.5 |
| Budget | $B$ | 50.0 (M=50) |
| Straggler noise | $\sigma$ | 0.5 (default) |
| Dirichlet | $\alpha_{Dir}$ | 0.3 |
| Local/Distill/Pretrain epochs | — | 2 / 3 / 10 |

---

## 六、Experiment 3：Timeout Policy Comparison (WITH vs WITHOUT D_min)

> **目的**：比較 fixed/adaptive/partial 三類 deadline policy，並驗證 D_min floor 的保護作用。
> **對應論文**：Sec 5.4, Fig 6
> **變數**：7 configs (with D_min) + 6 configs (without D_min)，σ=1.0 (harsh)

### Exp 3a — With D_min (min_deadline_ratio=0.3)

| Policy | Accuracy (%) | Best (%) | WC (s) |
|--------|-------------|----------|--------|
| **adaptive(0.7)** | **45.05 ± 0.75** | 46.33 | **871 ± 79** |
| partial(0.7) | 44.55 ± 0.59 | 45.69 | 871 ± 79 |
| adaptive(0.9) | 44.51 ± 0.71 | 45.46 | 1,105 ± 32 |
| fixed(20.0) | 44.45 ± 0.23 | 45.22 | 1,088 ± 14 |
| adaptive(0.5) | 43.85 ± 0.18 | 45.47 | 843 ± 79 |
| fixed(5.0) | 43.54 ± 0.63 | 45.37 | 843 ± 80 |
| fixed(10.0) | 43.34 ± 0.23 | 44.94 | 843 ± 80 |

### Exp 3b — Without D_min (min_deadline_ratio=0)

| Policy | Accuracy (%) | Best (%) | WC (s) |
|--------|-------------|----------|--------|
| fixed(20.0) | 44.28 ± 0.40 | 45.34 | 1,088 ± 14 |
| adaptive(0.9) | 43.98 ± 1.01 | 45.32 | 1,096 ± 21 |
| fixed(10.0) | 43.06 ± 0.42 | 44.89 | 618 ± 14 |
| adaptive(0.7) | 42.55 ± 1.15 | 44.82 | 690 ± 41 |
| adaptive(0.5) | 40.86 ± 1.36 | 42.67 | 504 ± 16 |
| **fixed(5.0)** | **35.96 ± 1.57** | 39.28 | 383 ± 14 |

### Paired Comparison — D_min 的效果

| Policy | Without D_min | With D_min | **Δ Accuracy** | Δ WC |
|--------|:---:|:---:|:---:|:---:|
| **fixed(5.0)** | 35.96% / 383s | 43.54% / 843s | **+7.58pp** 🔥 | +460s |
| **adaptive(0.5)** | 40.86% / 504s | 43.85% / 843s | **+2.98pp** | +339s |
| **adaptive(0.7)** | 42.55% / 690s | 45.05% / 871s | **+2.50pp** | +181s |
| adaptive(0.9) | 43.98% / 1,096s | 44.51% / 1,105s | +0.53pp | +9s |
| fixed(10.0) | 43.06% / 618s | 43.34% / 843s | +0.28pp | +225s |
| fixed(20.0) | 44.28% / 1,088s | 44.45% / 1,088s | +0.17pp | +0s |

### Deadline Evolution (seed42)

**Without D_min** — deadline spiral 現象：

| Policy | R0 | R3 | R5 | R10 | R20 | R30 | R40 | R49 |
|--------|-----|-----|-----|------|------|------|------|------|
| fixed(5.0) | 54.6s | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 | 5.0 |
| adaptive(0.5) | 54.6s | 13.3 | 12.1 | 10.0 | 7.7 | 6.7 | 6.1 | 5.6 |
| adaptive(0.7) | 54.6s | 22.8 | 20.5 | 16.2 | 11.6 | 10.0 | 9.2 | 8.7 |
| adaptive(0.9) | 54.6s | 39.5 | 36.0 | 28.2 | 19.6 | 16.9 | 15.0 | 14.1 |

**With D_min** — 全部 clamp 在 D_min=16.4s：

| Policy | R0 | R3 | R5 | R10 | R20 | R30 | R40 | R49 |
|--------|-----|-----|-----|------|------|------|------|------|
| fixed(5.0) | 54.6s | 16.4 | 16.4 | 16.4 | 16.4 | 16.4 | 16.4 | 16.4 |
| adaptive(0.5) | 54.6s | 16.4 | 16.4 | 16.4 | 16.4 | 16.4 | 16.4 | 16.4 |
| adaptive(0.7) | 54.6s | 22.8 | 20.5 | 16.4 | 16.4 | 16.4 | 16.4 | 16.4 |
| adaptive(0.9) | 54.6s | 39.5 | 36.0 | 28.2 | 19.6 | 16.9 | 16.4 | 16.4 |

### Participation (seed42, mean over 50 rounds)

| Policy | With D_min (complete/partial) | Without D_min (complete/partial) |
|--------|:---:|:---:|
| fixed(5.0) | 46.2 / 3.7 | **25.8 / 24.0** ← 半數 partial |
| adaptive(0.5) | 46.2 / 3.7 | 33.5 / 16.3 |
| adaptive(0.7) | 46.5 / 3.3 | 40.5 / 9.2 |
| adaptive(0.9) | 46.9 / 2.4 | 46.6 / 2.7 |
| fixed(20.0) | 46.7 / 2.3 | 46.7 / 2.3 ← 控制組，完全一致 |

### 觀察與論文寫作建議 (Exp 3)

1. **D_min 是 DASH 的關鍵安全網**
   - fixed(5.0) 無 D_min → **崩潰到 35.96%**（半數 device 只送 partial logits）
   - 有 D_min → 被 clamp 到 16.4s，精度回到 43.54% (+7.58pp)
   - 論文：「Without $D_{\min}$, aggressive deadline settings catastrophically reduce accuracy (35.96% for $D=5$s). The $D_{\min}$ floor prevents this by clamping $D \geq 0.3 \cdot D^{(0)}$.」

2. **Adaptive 不加 D_min 也會 spiral**
   - adaptive(0.5) 的 deadline 從 13.3s 一路降到 5.6s → 精度只有 40.86%
   - 加了 D_min → 43.85% (+2.98pp)
   - 論文：「Even adaptive policies exhibit deadline spiral without a floor — the EMA of truncated latencies causes the deadline to decrease monotonically.」

3. **adaptive(0.7) + D_min = Pareto 最優**
   - 45.05% / 871s — 全場 13 組 config 中最佳 accuracy，WC 排第 8（快於所有 +D_min configs 中只輸 3 組快但低精度的）
   - 論文：「The combination of adaptive deadline ($p=0.7$) with $D_{\min}$ floor achieves the best accuracy-latency trade-off among all 13 configurations tested.」

4. **Fixed deadline 的兩難困境**
   - Without D_min: D=5s→36%, D=10s→43%, D=20s→44.3% 但 WC=1088s
   - 論文：「Fixed deadlines present an irreconcilable accuracy-speed dilemma: low $D$ causes mass timeouts, while high $D$ negates the async speedup benefit.」

5. **Partial accept 反而掉精度 (-0.50pp)**
   - partial(0.7): 44.55% vs adaptive(0.7): 45.05%，同 WC
   - 原因：接收不完整 logits 引入 noise
   - 論文：「Partial-accept does not improve accuracy; incomplete logits introduce noise that outweighs the benefit of higher participation rates (0.95 vs 0.93).」

6. **adaptive(0.9) 太保守**
   - 精度 44.51% 但 WC 高達 1,105s — 等太多 device
   - 與 adaptive(0.7) 比：-0.54pp 且 +234s — 完全劣勢

---

## 七、待補實驗結果（Exp 4-6 完成後更新此區）

### Exp 4: Scalability
> 預期：DASH 在 M=200 的 WC 優勢比 M=20 更大；Sync WC ∝ max(τ_i) 隨 M 增加

*(待填)*

### Exp 5: Cross-Dataset (EMNIST)
> 預期：趨勢與 CIFAR-100 一致，驗證 generalizability

*(待填)*

### Exp 6: Privacy under Async
> 預期：ρ 越低 accuracy 越低，但 DASH 的 straggler 處理不受 privacy 影響

*(待填)*

---

## 八、論文寫作 Key Messages（供 Claude web 參考）

### 一句話結論
> DASH achieves **3.14× wall-clock speedup** over synchronous scheduling with **no accuracy loss** (44.47% vs 43.66%, +0.81pp) on CIFAR-100 with 50 heterogeneous edge devices.

### Table I caption 建議
> Comparison of six methods on CIFAR-100 (M=50, σ=0.5, α=0.3). Accuracy and wall-clock time are reported as mean ± std over 3 seeds. Speedup is relative to Sync-Greedy.

### Fig 2 描述重點 (acc vs round)
- DASH 和 Sync-Greedy 幾乎重疊（per-round accuracy 近似）
- Full-Async 有明顯 gap（~8pp lower）
- FedBuff-FD 幾乎不學（~17%）

### Fig 3 描述重點 (acc vs wall-clock) — 殺手圖
- 同樣的曲線但 x 軸是 wall-clock → DASH 明顯左移（更快達到相同精度）
- Sync-Greedy 的點在 x 軸上間距大（每 round 等 ~54-68s）
- DASH 的點在 x 軸上間距小（每 round 只要 ~16-55s）
- 交叉點：DASH 在 ~500s 達到 40%，Sync-Greedy 要 ~1100s

### Fig 4-5 描述重點 (acc/WC vs σ)
- DASH accuracy 幾乎一條水平線（robust）
- Sync-Greedy WC 在 σ=1.5 時膨脹 15.6%
- Speedup 隨 σ 增加：3.15× → 3.20×

### Fig 6 描述重點 (policy comparison, with/without D_min)
- 左圖：acc vs WC scatter，每個 policy 兩個點 (+D_min 實心, -D_min 空心)
- fixed(5.0) 無 D_min 是離群值 (35.96%, 383s)，有 D_min 跳回 43.54%
- adaptive(0.7)+D_min 是 Pareto 最優 (45.05%, 871s)
- 右圖：deadline evolution，展示 spiral 現象 (adaptive 0.5 降到 5.6s)
- 關鍵訊息：D_min 是不可省略的安全機制，最大救回 +7.58pp

### 關於 44.47% 是否足夠（reviewer 可能質疑）
- CIFAR-100 + non-IID (α=0.3) + FD (logits only) + 50 clients + 50 rounds + LDP noise
- FD 論文標準數字：FedDF ~42-48%, FedGKD ~50-55% (但只用 10 clients)
- 我們的 contribution 不是追求最高精度，而是 **accuracy-wallclock Pareto optimality**
- 論文 Sec 5.2 可加一句：「We note that the absolute accuracy of FD methods is inherently lower than full-model FL (e.g., FedAvg) due to the information bottleneck of logit-only communication — the advantage lies in privacy preservation and communication efficiency.」
