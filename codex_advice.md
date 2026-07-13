› 评估一下这个发AAAI /home/dy23a.fsu/popllm/SpatialMind/RESULTS_v10.md


• 我会先通读 RESULTS_v10.md，再结合仓库里的论文定位、方法和实验上下文，按 AAAI 的创新性、技术质量、实验充分性与写作风险给出审稿式评估。

• 初步看，结果有一个明显的亮点，也有一个会被审稿人重点攻击的点：spartqa 上两个弱基方法融合到 0.721 很醒目，但“每个目标数据集用带标签 validation 选择模式并拟合 combiner”不应被表述成一般
  OOD/transfer，且需要多随机种子和置信区间证明不是小验证集上的选择收益。接下来我会核对 fusion 实现、标签定义和基线训练协议。

• 实现确认了它没有读取 test 标签来训练，但论文层面的风险不等于代码泄漏：这是“target-domain supervised adaptation”，因为每个测试域都使用了 196–1000 个带标签样本训练最终逻辑回归；因此
  fusion 与直接迁移的 heads 并非同一协议，不能放在一张表里暗示同类比较。我再计算标签比例、融合选型和 bootstrap 区间，判断 headline 是否统计稳健。

• ## 总体判断

  以当前 RESULTS_v10.md 的证据直接投 AAAI，我会给 Weak Reject，录用概率约 20%–30%。

  如果把实验协议和统计证据补齐，有机会提升到 Borderline / Weak Accept，约 40%–55%。目前不是结果不够亮眼，而是核心贡献的表述与实验协议存在错位，容易被审稿人抓住。

  ## 主要亮点

  1. StepGame 上约 30 点 AUROC 的优势很强
      - constraint 方法达到 0.84–0.87。
      - neural/statistical UQ 基线只有 0.46–0.57。
      - 零参数 constraint_rule 达到 0.789，说明增益确实来自显式空间约束，而不只是参数量。

  2. 跨 6 个数据集、约 2k 测试样本的规模基本达到主会实验要求
      - 比 v9 的小规模结果可信很多。
      - leakage audit 和 test-label permutation 检查是明显加分项。

  3. Fusion 的效果具有统计意义

     我对保存的预测做了 300 次 paired bootstrap，近似 95% CI：

      数据集        Fusion AUROC 95% CI    相对最佳纯方法差值 CI
     ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━
      StepGame           [0.845, 0.885]         [-0.010, -0.002]
     ────────────  ─────────────────────  ───────────────────────
      spartqa            [0.693, 0.747]         [+0.237, +0.347]
     ────────────  ─────────────────────  ───────────────────────
      babi               [0.625, 0.672]         [+0.044, +0.114]
     ────────────  ─────────────────────  ───────────────────────
      SpaRTUN            [0.620, 0.664]         [+0.031, +0.067]
     ────────────  ─────────────────────  ───────────────────────
      SpaceNLI           [0.752, 0.795]         [+0.021, +0.060]
     ────────────  ─────────────────────  ───────────────────────
      SpartQA_YN         [0.479, 0.526]         [-0.057, +0.019]

     spartqa、babi、SpaRTUN、SpaceNLI 的提升不是普通采样噪声；尤其 spartqa 的结果很值得深入分析。

  ## 致命风险

  ### 1. Fusion 实际是目标域有监督适配，不是通常意义的 OOD transfer

  Fusion 会在每个目标数据集上使用 196–1000 个带标签 validation 样本：

  - 选择 scores_only/symb/determinacy 和 L2；
  - 重新用整个 target validation 拟合 logistic regression；
  - 然后测试目标域。

  实现见 scripts/gated_fusion.py。

  这确实没有 test leakage，但它属于：

  > few-shot target-domain supervised adaptation / domain-specific stacking

  而不是零样本 OOD，也不应与只在 StepGame 训练后直接迁移的 heads 放在同一协议下比较。审稿人很可能会说：

  > 跨域提升可能来自目标域标签，而不是提出的 symbolizability routing。

  尤其 spartqa 上，从两个低于或接近随机的分数得到 0.721，并不神奇：target validation 上的逻辑回归可以学习重新定向、组合甚至反转这些分数。

  必须增加：

  - 不使用目标标签的 zero-shot fusion；
  - 只在 StepGame validation 训练一个统一 gate，冻结后应用全部 OOD；
  - leave-one-dataset-out gate；
  - target-label budget 曲线：0、16、32、64、128、完整 validation；
  - 同协议的监督适配基线，例如 target-val 上训练 MLP、temperature/logistic stacking、只用两个 score 的 stacking。

  否则 headline 应改成：

  > Lightweight target-domain adaptation combines complementary symbolic and neural uncertainty signals.

  而不能主打一般性的 OOD routing。

  ### 2. “Symbolizability-gated”贡献尚未被隔离证明

  当前 Fusion 同时做了：

  - 两个基础分数的 logistic stacking；
  - 分数乘积；
  - parse/determinacy 特征；
  - 多个交互项；
  - 每数据集模式和 L2 选择。

  因此 0.721 到底来自：

  - 普通 stacking；
  - 分数方向修正；
  - target-domain label fitting；
  - 还是 symbolizability/determinacy gate，

  目前无法判断。

  需要在主表加入严格消融：

   方法                                 目标
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━
   constraint only                      基础信号
  ───────────────────────────────────  ────────────────────────
   neural only                          基础信号
  ───────────────────────────────────  ────────────────────────
   fixed 0.5 average                    无学习融合
  ───────────────────────────────────  ────────────────────────
   target-val global linear stacking    测试监督 stacking 本身
  ───────────────────────────────────  ────────────────────────
   scores-only logistic                 现有代码已有
  ───────────────────────────────────  ────────────────────────
   + parse rate                         验证 symbolizability
  ───────────────────────────────────  ────────────────────────
   + determinacy                        验证核心新贡献
  ───────────────────────────────────  ────────────────────────
   full features                        容量控制

  最关键的是报告 scores-only 与 determinacy 的 paired 显著性差异。否则论文贡献可能会被归类为“在验证集训练一个 logistic regression”。

  ### 3. 核心方法没有胜过自己的 ablation

  StepGame：

  - spatialmind main: 0.871
  - constraint_no_conflict: 0.874
  - constraint_only: 0.847
  - Fusion 最终还选择了 constraint_no_conflict，而不是 main。

  这会直接触发审稿问题：

  > 为什么删除 conflict 模块反而更好？为什么最终系统使用一个命名为 ablation 的模型？

  虽然差异可能不显著，但当前证据说明 conflict 模块没有正贡献。更严重的是，多个 OOD 数据集上 main 也明显差于 constraint_only：

  - spartqa：0.343 vs 0.496
  - SpaRTUN：0.420 vs 0.547

  这削弱了“神经与约束联合 head”的设计合理性。

  建议重新定义正式方法：

  - 如果 constraint_no_conflict 最稳定，就把它变成正式的 constraint scorer；
  - 将原 spatialmind 降为一种 learned hybrid baseline；
  - 不要让 headline 系统依赖一个事后选中的 ablation。

  ### 4. 缺少多随机种子

  当前 v10 看起来是单个生成 seed、单个训练 seed和一次目标验证划分。仓库虽然有 jobs/multiseed.sh，但结果文件没有多 seed 结果。

  AAAI 审稿人会特别关心：

  - neural heads 的训练方差；
  - validation even/odd 拆分的选择方差；
  - 生成采样方差；
  - target validation 规模较小时的 gate 方差，尤其 bAbI validation 只有 196。

  最低要求：

  - 5 个训练 seed；
  - 5–10 个 validation split seed；
  - 报告 mean ± std；
  - test prediction 上 paired bootstrap 或 DeLong CI。

  当前 bootstrap 只证明固定模型的 test sampling uncertainty，不替代训练方差。

  ## 其他重要问题

  ### LLM-as-judge 可靠性

  trace correctness 是核心标签，但 bAbI 等数据依赖 24B judge。需要至少：

  - 200–500 条人工双人标注；
  - judge 与人工的一致率、F1、Cohen’s κ；
  - 分数据集报告；
  - 检查 judge 是否系统性偏向短答案、格式或 backbone 风格。

  否则方法可能是在预测 judge 行为，而不是 hallucination correctness。

  ### 缺少真正强且近期的基线

  当前部分基线名称足够多，但论文需要明确：

  - 是否使用官方实现；
  - 输入特征和监督预算是否公平；
  - 每个方法是否同样获得 target validation labels；
  - 是否包含 semantic consistency/self-consistency、verbalized confidence、训练式 correctness predictor、selective prediction 等强基线。

  同协议公平性比“基线数量”更重要。

  ### 只用一个 backbone

  README 提到多个 backbone，但 v10 headline 只有 Llama-3.1-8B。AAA​​I 对“LLM hallucination/UQ 方法”的可泛化性要求通常较高。

  建议至少加入：

  - 一个不同家族开源模型；
  - 一个更强或更大模型；
  - 最好验证跨 backbone 训练与迁移。

  ### SpartQA_YN 没有形成正面证据

  所有方法接近随机，Fusion CI 也跨过 0.5。这是诚实的负结果，但现在它没有验证 symbolizability thesis，反而暴露 solver 对 multi-hop grounding 的限制。

  应把它转化为：

  - coverage/determinacy failure analysis；
  - 按 hop 数、unknown rate、parse rate 分桶；
  - 展示可预测的 applicability boundary。

  ### 文本存在数字和措辞问题

  RESULTS_v10.md 中：

  - 表格 mean 是 0.693，Finding 2 写成 0.690。
  - “Core novelty holds”过强，结果支持 constraint signal 有效，但尚未证明完整方法的新颖性。
  - “never collapses to the worse method”最好删除或严格定义。
  - “both pure methods near-useless yet fusion recovers 0.721”容易引起怀疑，应明确这是 target-supervised stacking，并解释分数条件互补或方向反转机制。
  - SpartQA_YN 是 machine-generated 数据集，需要详细说明生成流程、去重、难度控制和是否公开。

  ## 推荐论文定位

  当前最稳妥的论文故事不是：

  > 一个通用的 OOD hallucination detector。

  而是：

  > Explicit spatial constraint execution provides a complementary uncertainty signal, whose usefulness depends on semantic determinacy; a lightweight applicability-aware adapter
  > combines it with neural uncertainty under limited target supervision.

  对应的三项贡献可以写成：

  1. 将生成 reasoning trace 转成可执行空间约束，并提取 contradiction、entailment、repair、prefix consistency 信号。
  2. 系统揭示 constraint UQ 的适用边界由 determinacy 而非单纯 parse coverage 决定。
  3. 提出 applicability-aware adaptation，在有限目标域标注下稳定组合 symbolic 和 neural UQ。

  这个定位与当前实验事实更一致，也更难被协议问题击穿。

  ## 投稿前优先级

  必须补：

  1. Zero-shot、target-adapted 两套协议彻底分表。
  2. scores-only stacking 与 determinacy gate 的严格消融。
  3. 多 seed、置信区间和 validation split 稳定性。
  4. judge 人工一致性评估。
  5. 至少第二个 backbone。
  6. 统一正式方法，解决 main 不如 no_conflict 的问题。
  7. target-label budget curve 和同预算适配基线。

  **结论：**当前结果已经足以支撑一篇有潜力的论文，但还不足以支撑 AAAI 主会的强结论。最大问题不是 AUROC，而是 Fusion 的收益尚未与 target-domain supervision 和普通 stacking 的收益解耦。只
  要这点补扎实，spartqa 上的大幅提升会从“可疑结果”变成论文最有价值的发现。


› 评估一下这个发AAAI /home/dy23a.fsu/popllm/SpatialMind/RESULTS_v10.md


• 我会先通读 RESULTS_v10.md，再结合仓库里的论文定位、方法和实验上下文，按 AAAI 的创新性、技术质量、实验充分性与写作风险给出审稿式评估。

• 初步看，结果有一个明显的亮点，也有一个会被审稿人重点攻击的点：spartqa 上两个弱基方法融合到 0.721 很醒目，但“每个目标数据集用带标签 validation 选择模式并拟合 combiner”不应被表述成一般
  OOD/transfer，且需要多随机种子和置信区间证明不是小验证集上的选择收益。接下来我会核对 fusion 实现、标签定义和基线训练协议。

• 实现确认了它没有读取 test 标签来训练，但论文层面的风险不等于代码泄漏：这是“target-domain supervised adaptation”，因为每个测试域都使用了 196–1000 个带标签样本训练最终逻辑回归；因此
  fusion 与直接迁移的 heads 并非同一协议，不能放在一张表里暗示同类比较。我再计算标签比例、融合选型和 bootstrap 区间，判断 headline 是否统计稳健。

• ## 总体判断

  以当前 RESULTS_v10.md 的证据直接投 AAAI，我会给 Weak Reject，录用概率约 20%–30%。

  如果把实验协议和统计证据补齐，有机会提升到 Borderline / Weak Accept，约 40%–55%。目前不是结果不够亮眼，而是核心贡献的表述与实验协议存在错位，容易被审稿人抓住。

  ## 主要亮点

  1. StepGame 上约 30 点 AUROC 的优势很强
      - constraint 方法达到 0.84–0.87。
      - neural/statistical UQ 基线只有 0.46–0.57。
      - 零参数 constraint_rule 达到 0.789，说明增益确实来自显式空间约束，而不只是参数量。

  2. 跨 6 个数据集、约 2k 测试样本的规模基本达到主会实验要求
      - 比 v9 的小规模结果可信很多。
      - leakage audit 和 test-label permutation 检查是明显加分项。

  3. Fusion 的效果具有统计意义

     我对保存的预测做了 300 次 paired bootstrap，近似 95% CI：