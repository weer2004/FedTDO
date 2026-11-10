# FedTDO ICML 2026 manuscript — reproducibility audit

审计基准：GitHub 仓库 weer2004/FedTDO 的 main 分支（本次撰稿时读取）。论文草稿写在同一分支副本的 paper/fedtdo_icml2026.tex；本文件列出方法描述、用户提供命令与代码实现之间的差异。**不要在这些问题解决前把草稿中的结果表填成正式结果，也不要把当前 first 模式称为严格因果的在线证据更新。**

## 1. 当前命令无法在 main 上直接运行

run_fedtdo.py 使用严格的 argparse.parse_args()。用户给出的命令有以下不兼容项：

| 用户命令 | 当前 main 支持情况 | 应对 |
|---|---|---|
| --fedtdo_contdol fedtdo_nofreeze | 不存在该参数 | 参数名是 --fedtdo_control |
| --fedtdo_evidence_alpha 0.05 | 不存在 | 删除，除非先实现并验证相应检验 |
| --fedtdo_evidence_hac_lag 0 | 不存在 | 删除，当前没有 HAC/Newey–West 方差估计 |
| --fedtdo_rank_stat tail | 不存在 | 删除；当前部署排序使用 --fedtdo_estimator 对应的估计量，默认 mean |

代码位置：

- 参数解析与严格解析：[run_fedtdo.py 参数区](https://github.com/weer2004/FedTDO/blob/main/code/run_fedtdo.py#L14-L20)、[选项定义](https://github.com/weer2004/FedTDO/blob/main/code/run_fedtdo.py#L209-L220)、[统计选项](https://github.com/weer2004/FedTDO/blob/main/code/run_fedtdo.py#L258-L276)、[parse_args](https://github.com/weer2004/FedTDO/blob/main/code/run_fedtdo.py#L340-L350)
- 控制器只声明 leader / ranked 两种 gate：[fedtdo_controller.py](https://github.com/weer2004/FedTDO/blob/main/code/federated/fedtdo_controller.py#L13-L18)
- 当前 evidence gate 的统计量：[evidence check](https://github.com/weer2004/FedTDO/blob/main/code/federated/fedtdo_controller.py#L465-L477)

因此，alpha、HAC、rank-stat 参数不能被描述为“虽然 main 不支持但会被忽略”；它们会让命令行解析失败。

## 2. ranked 的实际统计含义

当前 --fedtdo_decision evidence --fedtdo_evidence_gate ranked 的行为是：

1. 按部署分数对已观察候选排序；
2. 逐个检查候选最近 evidence_k 条已记录增益；
3. 计算普通样本方差下的单样本 t 统计量；
4. 候选需要有正的平均增益，且 t 统计量不小于 evidence_t；
5. 选择排序中第一个通过者，否则回退到本地模型。

对所给的 k=4, t=1，它是**固定 t-statistic 阈值的顺序可行性门控**，不是 p-value 为 0.05 的检验。当前 gate 没有 Holm/BH 分支，不提供 FWER 控制；也没有 HAC lag 参数或 Newey–West 标准误。

还有一个实现细节要准确说明：部署排序分数调用 estimate()，其默认 estimator 是按 age/drift 权重衰减的 mean。_evidence_passes() 则取最近 k 条 observation 的原始 gain 并计算普通样本 t 统计量，没有应用 _weighted_history() 返回的权重。因此，当前实现不是“tail-stat 排序 + HAC 检验”。

相关位置：

- gate 与统计量：[fedtdo_controller.py](https://github.com/weer2004/FedTDO/blob/main/code/federated/fedtdo_controller.py#L465-L531)
- age/drift 衰减：[fedtdo_controller.py](https://github.com/weer2004/FedTDO/blob/main/code/federated/fedtdo_controller.py#L244-L251)
- 默认 estimator 及可选 estimator：[run_fedtdo.py](https://github.com/weer2004/FedTDO/blob/main/code/run_fedtdo.py#L258-L276)、[selector.py](https://github.com/weer2004/FedTDO/blob/main/code/federated/selector.py#L44-L45)

## 3. fedtdo_nofreeze 不是 canonical FedTDO 主臂

当前 parser 将：

- fedtdo 定义为 shared PatchEmbedding warm-up + FedAvg + freeze；
- fedtdo_nofreeze 定义为 FedAvg 后仍允许客户端继续训练共享 encoder 的诊断变体。

Exp_FedTDO 对 fedtdo_nofreeze 明确跳过冻结。因此，若主结果运行值确实是 fedtdo_nofreeze，论文必须把它命名为 no-freeze 变体，不应同时写成“共享 encoder 固定、capsule 表征空间保持一致”的 canonical 方法。需要冻结版本作为主方法时，使用：

~~~bash
--fedtdo_control fedtdo
~~~

若要报告 no-freeze，应单独设置一行 ablation/control。

代码位置：

- 控制定义：[run_fedtdo.py](https://github.com/weer2004/FedTDO/blob/main/code/run_fedtdo.py#L74-L95)
- 冻结逻辑与 no-freeze 分支：[exp_fedtdo.py](https://github.com/weer2004/FedTDO/blob/main/code/exp/exp_fedtdo.py#L477-L499)

## 4. FedTDO 与 baseline 的训练/聚合预算并不等价

给出的 FedTDO epoch 配置：

- fedtdo_encoder_rounds=1
- fedtdo_warmup_epochs=5
- fedtdo_epochs=60
- fedtdo_step_mode=epochs

在当前 Exp_FedTDO 中，实际含义是：客户端先进行一轮 5-epoch 的共享模块 warm-up，然后执行一次 synchronize_modules()；之后进行最多 60 个 downstream 本地 epoch（由 validation patience 早停）。因此共享 PatchEmbedding 的同步是一次，不是五次。

给出的 FedProx baseline：

- fl_rounds=5
- fl_local_epochs=12

代表五轮联邦训练、每轮每客户端 12 个 local epochs，总计 60 个本地 epoch，并在这些轮次中执行多次聚合。两个配置可能具有相近的 local epoch 数，但**通信/聚合时序不同**，不能概括成“同一协议”或“5 轮 × 12 epochs 的 FedTDO”。

代码位置：

- FedTDO warm-up / synchronize / freeze：[exp_fedtdo.py](https://github.com/weer2004/FedTDO/blob/main/code/exp/exp_fedtdo.py#L383-L499)
- FedTDO downstream epochs：[exp_fedtdo.py](https://github.com/weer2004/FedTDO/blob/main/code/exp/exp_fedtdo.py#L502-L535)
- Baseline 轮数与本地 epoch 参数：[run_baselines.py](https://github.com/weer2004/FedTDO/blob/main/code/run_baselines.py#L121-L142)

论文应报告准确的聚合时序，而不是只比较“总共 60 epochs”。

## 5. first 反馈当前存在全预测区间信息泄漏

这是当前最重要的实验有效性问题。

--fedtdo_online_target first 在 Exp_FedTDO.run() 中把反馈释放时间设为下一预测 origin；_online_step() 也确实将 SGD target 切成第一步。但随后代码调用 client.reveal_feedback(record_id, old_target) 时传入的是**完整预测区间的 target**。

接下来：

- PrequentialBuffer.resolve() 用完整 target 计算 local loss、mixed candidate gains 和 pure expert gains；
- FedTDOClient._update_alpha() 也使用完整 target 更新 expert mix-weight 的最小二乘统计量；
- 这些完整 horizon gains 随后被送入 controller history，进而影响之后的候选排序和 evidence gate。

如果 pred_len=96，在只释放第一个标签时，其余 95 个未来标签仍会影响迁移决策；ILI 的 H=24/36/48/60 同理。因而当前代码虽然把**梯度更新**截成第一步，却没有把**控制器反馈、候选评分和 alpha calibration**截成第一步。此结果不满足严格的 first-step prequential causality。

代码位置：

- first 对应 prefix=1 与 delay：[exp_fedtdo.py](https://github.com/weer2004/FedTDO/blob/main/code/exp/exp_fedtdo.py#L607-L617)
- 反馈释放时先在线更新、再以完整 old_target 调 reveal_feedback：[exp_fedtdo.py](https://github.com/weer2004/FedTDO/blob/main/code/exp/exp_fedtdo.py#L647-L657)
- SGD 只截 prefix：[exp_fedtdo.py](https://github.com/weer2004/FedTDO/blob/main/code/exp/exp_fedtdo.py#L789-L845)
- 完整 horizon loss/gain：[prequential.py](https://github.com/weer2004/FedTDO/blob/main/code/federated/prequential.py#L55-L86)
- alpha calibration 与 controller feedback：[client.py](https://github.com/weer2004/FedTDO/blob/main/code/federated/client.py#L316-L405)

**修复要求（二选一）：**

- 保留 first：在标签释放处统一裁切 target、local prediction、candidate predictions、external predictions；SGD、gain/loss 计算、alpha calibration、controller history 和 regime memory 全部只使用已释放的首步。完整 horizon 只能在未来全部标签到达后用于离线指标，不能提前反馈给控制器。
- 改用 full：将反馈延迟设置为完整 H 步；在完整 horizon 可观测之前，不更新 SGD、候选增益、alpha 校准、控制器历史或 regime memory。

修复后必须重跑全部方法和 horizon，不能只在论文措辞中解释该问题。

## 6. eval_budget=4 不一定等于每窗总共四次专家 forward

select_probes() 最多返回四个 shadow probe，但 FedTDOClient.predict() 还会把 active/deployed expert 加入 evaluated set；若它不在 probe set 中，会额外计算其 counterfactual forecast。故每窗外部专家评估数可能达到 probe budget + 1。论文应区分：

- shadow probes per origin（上限 4）；
- deployed active expert evaluation；
- 实际总 expert forwards（从 recorder 汇总）。

代码位置：

- probe 选择：[fedtdo_controller.py](https://github.com/weer2004/FedTDO/blob/main/code/federated/fedtdo_controller.py#L414-L463)
- active capsule 加入 evaluated set：[client.py](https://github.com/weer2004/FedTDO/blob/main/code/federated/client.py#L249-L283)

## 7. 数据划分、通道选择与 online steps

- 当前 channel partition 的单变量客户端采用按时间顺序 70/10/20 划分；StandardScaler 只在训练片段上 fit。验证和测试窗口保留前置 context。
- 当 ECL/Traffic 的 channel 数大于 32 且使用默认 channel_sampling=random 时，代码以 seed 选取随机子集；它并非简单取前 32 列。若要前 32 列，需要 --fedtdo_channel_sampling head。
- fedtdo_online_steps=1920 是**每个客户端**在线 stream 的 batch 数上限。数据 loader 提前耗尽时实际数量更小。尤其 ILI 必须从 run summary 报告实际执行窗口，不能直接把 1920 写成实际窗口数。
- 给出的 baseline 命令是 ECL/32 clients，不是 ETTh1/7 clients；应在表格中按 dataset 和客户端协议分别报告。

代码位置：

- channel client split、scaler、sampling：[federated_data.py](https://github.com/weer2004/FedTDO/blob/main/code/data_provider/federated_data.py#L23-L82)、[federated_data.py](https://github.com/weer2004/FedTDO/blob/main/code/data_provider/federated_data.py#L178-L264)
- per-client stream cap：[exp_fedtdo.py](https://github.com/weer2004/FedTDO/blob/main/code/exp/exp_fedtdo.py#L624-L645)

## 8. 结果记录开关

当前 run_fedtdo.py 的 --paper_record 默认是 0。给出的 FedTDO 命令没有显式设置 --paper_record 1，而 baseline 命令有 --paper_record 1。因此两边的 paper-grade event logging 开关不对称。

如要用 scripts/paper/collect_results.py、validate_results.py 和 final_checks.py 建立论文级结果，应在 FedTDO 运行中显式添加：

~~~bash
--paper_record 1 \
--paper_results_dir ./fixed/results/paper_fedtdo
~~~

并保留 per-window、feedback、online update、candidate evaluation、transfer、communication、cost 记录。必须从原始日志重建表格；目前仓库快照未检索到可支持论文性能数字的实验结果，因此草稿中的数值位置保持 TBD。

## 9. 建议的下一版实验定义

建议主论文协议固定为：

1. 使用 --fedtdo_control fedtdo 作为 canonical 主方法；将 fedtdo_nofreeze 放入单独的 no-freeze ablation。
2. 使用当前支持的 --fedtdo_decision evidence --fedtdo_evidence_gate ranked --fedtdo_evidence_k 4 --fedtdo_evidence_t 1，不要添加未实现的 alpha/HAC/rank-stat flags。
3. 先修复 first 的全链路 prefix feedback；之后再执行 ETTh1-H96 与 ILI-H24/36/48/60。
4. --paper_record 1，记录每个客户端实际的 stream 长度、评估次数、transfer 次数、通信字节和运行时间。
5. 让 FedTDO 与 baseline 在数据切分、预测 origin 数、released target prefix、online SGD、学习率、梯度裁剪等方面对齐；如果聚合次数不同，明确作为方法差异而不是称为完全相同的 training protocol。
6. 所有数值都由原始日志脚本生成，并对 seed 做重复实验后再写均值/方差或置信区间。

## 10. 稿件和 ICML 模板说明

稿件 fedtdo_icml2026.tex 使用仓库中的 ICML 2026 style package，并在结果表保留 TBD。ICML 2026 初始投稿主文上限为 8 页（参考文献和附录另计），投稿必须匿名；正式提交前还需要确认作者信息、匿名化代码材料、最终 PDF 页数和官方格式检查。官方说明见 [ICML 2026 Author Instructions](https://icml.cc/Conferences/2026/AuthorInstructions)。


## 11. Newly identified: FedTDO and FedProx online feedback protocols are not matched

The supplied FedTDO command sets --fedtdo_online_target first. In Exp_FedTDO.run(), this schedules feedback after one origin (feedback_delay = 1), and _online_step() slices the target to its first coordinate. By contrast, the supplied baseline command uses --fl_online_lr 0.0001; the baseline parser's help text explicitly describes this as an SGD update on a stored full-horizon target after its final label arrives at origin t + pred_len. Thus, even if both commands use 1920 stream batches and the same nominal learning rate, the online adaptation information set and update timing differ:

- FedTDO: one-step target, next-origin feedback/update;
- FedProx: full-H target, delayed until the final horizon label arrives.

This is a material protocol confound. A comparison cannot be called "same online protocol" or attribute a performance difference solely to federated transfer. Either make both arms use the same released prefix and delay (including their controller/selection feedback), or report the different feedback regimes explicitly and add matched-protocol controls. Also verify that both arms count stream windows and gradient updates identically rather than assuming equal 1920 caps imply equal realized work.

Code references:
- FedTDO target mode and release scheduling: [exp_fedtdo.py](https://github.com/weer2004/FedTDO/blob/main/code/exp/exp_fedtdo.py#L607-L675)
- FedTDO prefix slicing: [exp_fedtdo.py](https://github.com/weer2004/FedTDO/blob/main/code/exp/exp_fedtdo.py#L789-L845)
- Baseline full-horizon online update contract: [run_baselines.py](https://github.com/weer2004/FedTDO/blob/main/code/run_baselines.py#L133-L139)

## 12. Newly identified: the current LaTeX source has mismatched display-math environments

A structural scan of paper/fedtdo_icml2026.tex finds display delimiters that cross environment boundaries. For example, the patch-embedding formula opens with \[ but closes with \end{equation}; other formulae open with \begin{equation} and close with \]. The aggregate counts happen to show eight equation begins and ends and five \[ / \] pairs, but the nesting/order is invalid. This is a LaTeX build blocker, not merely a style issue. Labels placed inside \[...\] also will not behave as numbered equation labels.

Before claiming the manuscript is buildable:
1. Convert each labelled display to a properly paired \begin{equation}...\end{equation} block.
2. Keep unlabelled displays consistently as \[...\].
3. Run latexmk (or the official ICML build sequence) with BibTeX until there are no undefined references, duplicate labels, or fatal errors.
4. Inspect the generated PDF and count the main-body pages separately from references/appendix.

The previous file checks only confirmed that the expected text files existed and could be read back from GitHub; they did not compile the TeX source. Do not describe the LaTeX as compile-verified.

## 13. Initial-submission anonymization and repository exposure

The manuscript source uses anonymous author macros, but the draft branch is inside the identifiable public repository weer2004/FedTDO, and the manuscript's audit explicitly points to the named main repository and line-level source URLs. ICML's double-blind policy says not to include non-anonymized public-repository links in the submission and expects supplementary code to be anonymized; an anonymous GitHub repository is an allowed route. The current branch is suitable as an internal drafting workspace, not as the anonymous code artifact to submit. Prepare a separate anonymized repository/archive and remove repository-identifying URLs, acknowledgements, and identifying metadata from the submitted PDF and supplementary material. See the official ICML 2026 Author Instructions.

## 14. Updated severity ordering

The issues that can invalidate the headline experimental claim or block submission are now:

1. Critical — first-prefix feedback leakage: future horizon labels affect candidate gains/controller feedback despite first-step release.
2. Critical — unmatched baseline feedback regime: FedTDO is first-step/next-origin while the supplied FedProx online update is full-horizon/delayed.
3. Critical — LaTeX build blocker: display-math environments are crossed/mismatched; no successful PDF build has been verified.
4. High — unsupported CLI options: the supplied command fails against current main because several flags are absent/renamed.
5. High — training/aggregation schedule mismatch: one-shot shared-module synchronization is not the baseline's repeated FedProx rounds.
6. High — anonymous submission packaging: the public named repository and source links cannot simply be carried into the double-blind submission.
7. High — missing experimental evidence: no verified raw results were found in the inspected snapshot; tables must remain placeholders until reruns and log validation.

These are separate problems; fixing one does not fix the others.
