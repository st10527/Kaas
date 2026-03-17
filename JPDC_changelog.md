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

## 待確認事項（實驗跑完後檢查）

- [ ] **Accuracy 趨勢**：DASH ≥ Sync-Greedy >> FedBuff-FD >> Random-Async
- [ ] **Wall-clock 趨勢**：DASH 在相同 wall-clock 下收斂更快（這需要修正 3 完成後才能驗證）
- [ ] **Straggler sweep**：σ 越大，DASH 相對 Sync-Greedy 的 wall-clock 優勢越明顯
- [ ] **Timeout policy**：Adaptive ≈ Partial-Accept > Fixed（accuracy），Fixed 最快（wall-clock per round）
- [ ] **Scalability**：DASH 在 M=200 時的 wall-clock 優勢比 M=20 更大
- [ ] **Privacy**：ρ 越低 accuracy 越低，但 DASH 的 straggler 處理不受 privacy 影響
- [ ] **EMNIST**：趨勢與 CIFAR-100 一致
