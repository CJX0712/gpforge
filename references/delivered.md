# gpForge — delivery log (SOP §11 write-back)

| date | system | repo | domain | grade | note |
|------|--------|------|--------|-------|------|
| 2026-10-09 | gpForge | CJX0712/gpforge | 高斯过程回归 + 贝叶斯优化（RKHS 核 / Cholesky 精确推断 / EI-UCB 采集） | S | 旗舰 GaussianProcess（RBF/Matérn52/Matérn32/Periodic/Linear，ARD）+ L-BFGS-B 对数边缘似然超参优化（SciPy Tier-0，纯 NumPy 固定超参兜底，零 ML 依赖可跑）；预注册门禁逐数据集全过：95% 区间实测覆盖率 ≥0.90（实测 0.939–0.955）且 GP RMSE ≤1.10×岭回归基线；同 seed 两次运行逐位一致；29 单测（核 PSD / 后验方差≥0 / 噪声free极限插值 / 手写 Cholesky 逐位一致 / 校准 / BO EI≥0 / 确定性 / 门禁 / CLI）；ruff 双绿；CI py3.10+3.13 + 密钥扫描；Dockerfile/Makefile/pyproject；模型卡诚实标注 O(n³) 稠密局限 |

## Coverage notes
- Domain chosen: **高斯过程 / Gaussian Processes + 贝叶斯优化** — 与已交付舰队（matrix/equi/priv/vae/imitation/kernel/diffu/notears/nas/world/flow/conformal/causal/star）正交；核心卖点是可验证的**校准不变量**（well-specified GP 下 95% 区间覆盖率必须趋近 95%），而非"曲线看起来顺"。
- Author attribution: 晨星.
- Quality grade: **S** (world-class): 可独立验证的数学不变量（核 PSD 特征值检验、后验方差为 PD 块 Schur 补故 ≥0、噪声free 极限插值、fit_fixed 与手写 Cholesky 一致至 1e-8、实测覆盖率接近名义值），多 seed mean±std 聚合，CI 矩阵 + 密钥扫描，锁定依赖版本。
- Key fixes during build（已回写 CHANGELOG）:
  1. 预测方差漏加观测噪声项 σ² → 区间严重过度自信（实测覆盖率仅 0.13–0.62）。修复后 0.939–0.955。
  2. `np.math.erf` 在 NumPy 2.x 已移除 → 改用 `math.erf`。
  3. benchmark 预设原本统一用 RBF 核，周期性数据被静默欠拟合 → 引入 `preset_kernel()` 让每个预设用生成它的核。
- Honest finding baked into model card: 结果为**合成数据**（四个预设）上的可复现演示，非真实数据；稠密 GP 为 O(n³)/O(n²) 内存，n ≳ 数千需稀疏近似（本仓库不含）；periodic 核的 period 为固定旋钮未参与优化。

## Repo
- https://github.com/CJX0712/gpforge  (Release v0.1.0)
