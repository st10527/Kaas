# JPDC 2026 — 修正記錄與待辦事項

> 每次對代碼或實驗做修正時，在這裡記錄原因和影響。  
> 實驗全部跑完後，根據這份清單一次性修改論文 `2026_JPDC.tex`。

---

## 修正 1：KaaS-Edge → Sync-Greedy（label 改名）

| 項目 | 內容 |
|------|------|
| **日期** | 2026-03-17 |
| **Commit** | `816e786` |
| **原因** | KaaS-Edge 論文（IEEE EDGE 2026）尚在審稿，不能在 JPDC 中當作已發表 baseline 引用。 |
| **代碼修改** | `scripts/run_jpdc_experiments.py` 中所有 `'KaaS-Edge'` label → `'Sync-Greedy'`（Exp 1, 2, 4, 5）。底層程式碼不變，僅改顯示名稱。 |
| **論文待改** | Sec 5 (Evaluation)：所有提到「KaaS-Edge」作為 baseline 的地方改為「Sync-Greedy」。描述為：*"the synchronous variant of DASH with $\pi_i = 1$ for all devices (i.e., $\tilde{\rho}_i = \rho_i$), which serves as a natural ablation baseline."* Table I 和所有 Figure 的 legend 都要對應改。 |

---

## 修正 2：ρ̃_i substitution（數學框架修正）

| 項目 | 內容 |
|------|------|
| **日期** | 2026-03-17 |
| **Commit** | `e910877` |
| **原因** | 論文 Eq.(15) 定義 $\tilde{\rho}_i = \pi_i(D) \cdot \rho_i$ 代入 water-filling 和 quality 函數。原始代碼是事後在 greedy loop 做 `marginal_gain *= pi_i`，結構不同：water-filling 沒用到 π_i → 分配量不受 straggler 影響。 |
| **代碼修改** | `src/scheduler/rads.py`：在 `schedule()` 開頭計算 `rho_tilde_i = pi_i * rho_i`，用於 efficiency index、water-filling KKT、quality 計算。移除 post-hoc `marginal_gain *= pi_i`。 |
| **論文待改** | 無需修改（代碼修正是為了對齊論文，不是反過來）。確認 Algorithm 2 的 `WaterFill(S', ..., {ρ̃_j}, δ)` 與代碼一致。 |

---

## 修正 3：Sync-Greedy 缺少同步 wall-clock 模擬

| 項目 | 內容 |
|------|------|
| **日期** | 2026-03-17 |
| **Commit** | *待完成* |
| **原因** | 同步方法 (Sync-Greedy, Sync-Full) 的 `wall_clock_time = 0`，因為它們不走 straggler model。但論文需要 accuracy vs wall-clock 圖來比較 sync 和 async。同步版的 round time 應為 $\max_{i \in S^{(t)}} \tau_i$（等最慢的人），這正是 async 要解決的問題。 |
| **代碼修改** | 在同步方法的 `run_round()` 中，用 StragglerModel 模擬每個被選設備的 latency，取 max 作為該 round 的 wall-clock time。累加到 `wall_clock_time`。 |
| **論文待改** | Sec 5.1 (Setup)：加一句 *"For synchronous baselines, the per-round wall-clock time is set to $\max_{i \in S^{(t)}} \tau_i$, reflecting the blocking nature of synchronous aggregation."* Fig 3 (accuracy vs wall-clock) 和 Fig 5 (wall-clock vs straggler ratio) 的 caption 也要提到這個定義。 |

---

## 修正 4：NumPy 2.4 deprecation warning

| 項目 | 內容 |
|------|------|
| **日期** | 2026-03-17 |
| **Commit** | `019934f` |
| **原因** | `torchvision` 讀 CIFAR pickle 時觸發 NumPy 2.4 的 `VisibleDeprecationWarning`（`dtype(): align` 參數格式變更）。不影響計算結果。 |
| **代碼修改** | `scripts/run_jpdc_experiments.py` 開頭加 `warnings.filterwarnings("ignore", ...)` 壓掉。 |
| **論文待改** | 無。 |

---

## 修正 5：DASH warmup deadline 過大 + adaptive deadline 過緊 + v_feasible 過度壓縮

| 項目 | 內容 |
|------|------|
| **日期** | 2026-03-17 |
| **Commit** | *見下方* |
| **原因** | Exp 1 (50 dev × 50 rnd) 結果顯示 DASH 35.2% 嚴重輸給 Sync-Greedy 44.3%。根因三個：(1) warmup deadline 用 `v_max*0.4*2.0=746s`，3 輪暖機燒掉 90% wall-clock (2239/2492s)；(2) adaptive percentile=0.7 使 deadline 從 746 跳崖到 18.8s；(3) RADS v_feasible margin=0.5 把中慢設備壓到 v=79–286，遠低於預算允許的 418。 |
| **代碼修改** | (A) `src/methods/dash.py` `_estimate_warmup_deadline()`: 改用 budget-based v 估計（而非 v_max*0.4），safety 2.0→1.5，p90→p85。D_warmup: 746→55s。(B) `DASHConfig.adaptive_percentile`: 0.7→0.85。D_adaptive: 18.8→36.9s。(C) `src/scheduler/rads.py`: v_feasible margin 0.5→0.8。v_feasible 從 286 升至 899（不再是 binding constraint）。 |
| **數值驗證** | 修正後：quality ratio DASH/Sync = 1.00x（修正前 0.57x），DASH WC ≈ 1900s vs Sync 2518s（1.33x speedup），warmup 3 輪 164s（修正前 2239s）。 |
| **論文待改** | Sec 4.2 或 Algorithm 1 的 D^(0) 描述若提到 warmup 公式需對應更新。Table II (parameter settings) 的 percentile 值從 0.7 改為 0.85。 |

---

## 修正 6：三個模擬 bug（sync-seed / FullAsync warmup / deadline spiral）

| 項目 | 內容 |
|------|------|
| **日期** | 2026-03-17 |
| **Commit** | *見下方* |
| **原因** | Exp 1 (50d×50r) 跑出 DASH 44.3%/896s vs Sync-Greedy 45.2%/2697s。三個問題：(1) Sync wall-clock 每輪恆定 53.9s（`_simulate_sync_wallclock` 每次用 seed=42 建 StragglerModel）；(2) Full-Async warmup D=1306s（`_estimate_warmup_deadline` 用 config.v_max=10000 而非 scheduler.v_max=100）；(3) DASH deadline 從 54.6→9.1s 持續下降（adaptive deadline feedback loop：tighter D → smaller v → shorter tau → even tighter D）。 |
| **代碼修改** | (1) `_simulate_sync_wallclock` 加 `round_idx` 參數，seed=42+round_idx。(2) `_estimate_warmup_deadline` 用 `self.scheduler.v_max` 取代 `self.config.v_max`。(3) `DASHConfig` 新增 `min_deadline_ratio=0.3`，D_min = 0.3 × D_warmup ≈ 16.4s；adaptive deadline 低於 D_min 時 clamp。 |
| **預期效果** | (1) Sync wall-clock 有合理隨機波動。(2) Full-Async warmup 從 1306→13s，3 輪暖機 3917→39s。(3) DASH deadline 穩定在 ~16s 而非持續下降到 9s，每輪 comm_mb 維持 ~7MB。 |
| **論文待改** | Algorithm 1: 加 D_min floor 描述。Sec 4.2: 提到 $D^{(t)} = \max(D_{\min}, \text{EMA}_{0.3}(p_{85}(\tau^{(t-1)})))$。Table II: 加 `min_deadline_ratio = 0.3` 參數。 |

---

## 待確認事項（實驗跑完後檢查）

- [ ] **Accuracy 趨勢**：DASH ≥ Sync-Greedy >> FedBuff-FD >> Random-Async
- [ ] **Wall-clock 趨勢**：DASH 在相同 wall-clock 下收斂更快（這需要修正 3 完成後才能驗證）
- [ ] **Straggler sweep**：σ 越大，DASH 相對 Sync-Greedy 的 wall-clock 優勢越明顯
- [ ] **Timeout policy**：Adaptive ≈ Partial-Accept > Fixed（accuracy），Fixed 最快（wall-clock per round）
- [ ] **Scalability**：DASH 在 M=200 時的 wall-clock 優勢比 M=20 更大
- [ ] **Privacy**：ρ 越低 accuracy 越低，但 DASH 的 straggler 處理不受 privacy 影響
- [ ] **EMNIST**：趨勢與 CIFAR-100 一致
