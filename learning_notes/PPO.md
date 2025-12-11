# DeepSeekMath: PPO 与 GRPO 深度解析

本文基于 DeepSeekMath 论文 (Shao et al., 2024)，详细解析 **Proximal Policy Optimization (PPO)** 与论文提出的 **Group Relative Policy Optimization (GRPO)**。我们将结合公式 (1)(2)(3)(4) 及 Figure 4，深入探讨它们在数学推理任务中的工作原理及差异。

---

## 1. 强化学习基础知识补充

在深入 PPO 和 GRPO 之前，我们需要理解几个核心概念，特别是用户提到的 **GAE**。

### 1.1 策略 (Policy) $\pi_\theta$

**策略模型（Actor）**是我们希望训练的语言模型。给定一个问题（State $s$），它生成回答（Action $a$）的概率分布 $\pi_\theta(a|s)$。

### 1.2 价值函数 (Value Function) $V(s)$

**价值函数**预测从当前状态 $s$ 开始，未来能获得的累积奖励期望值。在 PPO 中，这通常需要一个独立的 **Critic** 模型来训练。

### 1.3 优势函数 (Advantage Function) $A(s, a)$

**优势函数**衡量了“在状态 $s$ 下采取动作 $a$ 比平均情况好多少”。

$$A(s, a) = Q(s, a) - V(s)$$

其中 $Q(s, a)$ 是采取动作 $a$ 后的实际价值。如果 $A > 0$，说明这个动作比平均好，应该鼓励；反之则抑制。

### 1.4 GAE (Generalized Advantage Estimation)

**GAE** 是计算优势 $A$ 的一种经典方法，旨在平衡 **偏差 (Bias)** 和 **方差 (Variance)**。
在标准的 Actor-Critic 架构（如 PPO）中，我们需要计算 TD Error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$。

GAE 定义优势为这些 $\delta$ 的指数加权平均：

$$A_t^{GAE} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}$$

> **依赖性**：GAE 高度依赖于一个准确的价值函数 $V(s)$（Critic 模型）。如果 Critic 不准，GAE 估算的优势就会有很大偏差，导致训练不稳定。

---

## 2. Proximal Policy Optimization (PPO) 回顾

**PPO** 是目前 LLM RLHF（基于人类反馈的强化学习）中最主流的算法。DeepSeekMath 论文指出了 PPO 在数学推理任务中的局限性。

### 2.1 PPO 的核心公式 (1)

PPO 的目标是最大化以下目标函数（简化版）：

$$\mathcal{L}_{PPO}(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_{\theta_{old}}} \left[ \min \left( \frac{\pi_\theta(o|q)}{\pi_{\theta_{old}}(o|q)} A, \text{clip}\left(\frac{\pi_\theta(o|q)}{\pi_{\theta_{old}}(o|q)}, 1-\epsilon, 1+\epsilon\right) A \right) \right] \quad (1)$$

- $\frac{\pi_\theta}{\pi_{\theta_{old}}}$ (Ratio): 新策略与旧策略产生该回复的概率比率。
- $A$ (Advantage): 优势值，通常由 GAE 计算得出。
- `clip`: 截断操作，防止新策略 $\pi_\theta$ 偏离旧策略太远，保证训练的稳定性（Trust Region）。

### 2.2 PPO 的奖励设计与 KL 惩罚 (公式 2)

为了防止模型为了“讨好”奖励模型（Reward Model）而输出乱码或破坏语言能力，PPO 通常会在每一步的奖励中加入 **KL 散度惩罚**：

$$r_t = r_\phi(q, o_{\le t}) - \beta \log \frac{\pi_\theta(o_t | q, o_{<t})}{\pi_{ref}(o_t | q, o_{<t})} \quad (2)$$

- $r_\phi$: 奖励模型给出的原始分数。
- $\pi_{ref}$: 参考模型（通常是 SFT 模型），用于约束 $\pi_\theta$ 不要发生灾难性遗忘。
- $\beta$: KL 惩罚系数。

### 2.3 PPO 的痛点

结合 Figure 4 (上半部分)，我们可以看到 PPO 的架构非常重：

1.  **Actor Model** ($\pi_\theta$): 策略模型（训练中）。
2.  **Reference Model** ($\pi_{ref}$): 参考模型（冻结），用于计算 KL。
3.  **Reward Model** ($r_\phi$): 奖励模型（冻结），用于打分。
4.  **Value Model / Critic** ($V$): 价值模型（训练中），用于辅助计算 GAE。

**缺点**：
- 需要同时显存加载 4 个模型（如果是同等规模的 7B 模型，显存压力巨大）。
- 训练 Critic 模型很困难：在数学题这种“生成一长串步骤最后才给答案”的任务中，Critic 很难精准预测每一个 token 的价值。

---

## 3. Group Relative Policy Optimization (GRPO) 详解

为了解决 PPO 的显存占用大和 Critic 训练难的问题，DeepSeekMath 提出了 **GRPO**。

### 3.1 GRPO 的核心思想

**抛弃 Critic 模型**。不再通过训练一个神经网络来预测 $V(s)$ 作为 Baseline，而是通过采样一组输出（Group），计算这组输出的平均分作为 Baseline。

### 3.2 组采样与优势计算

对于同一个问题 $q$，GRPO 从旧策略 $\pi_{\theta_{old}}$ 中采样一组输出 $\{o_1, o_2, ..., o_G\}$。
通过奖励模型计算出分数 $\{r_1, r_2, ..., r_G\}$ 后，直接在组内计算 **相对优势**：

$$\hat{A}_{i,t} = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$$

如果一个回答的分数 $r_i$ 高于组内平均分，它的优势就是正的；反之则是负的。
这巧妙地利用了同一问题不同回答之间的对比，替代了 Critic 的作用。

### 3.3 GRPO 的目标函数 (公式 3)

GRPO 的目标函数如下：

$$\mathcal{J}_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left\{ \min \left( \frac{\pi_\theta}{\pi_{\theta_{old}}} \hat{A}_{i,t}, \text{clip}(\dots) \hat{A}_{i,t} \right) - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) \right\} \right] \quad (3)$$

**关键变化**：
1.  **没有 Critic**：公式中不再依赖由 $V(s)$ 计算出的 GAE，而是直接使用组相对优势 $\hat{A}_{i,t}$。
2.  **KL 正则化外移**：注意公式 (2) 中 PPO 将 KL 放在奖励 $r_t$ 里；而 GRPO (公式 3) 将 KL 散度 $\mathbb{D}_{KL}$ 直接作为一个正则项加在 Loss 函数中。这样做的好处是计算优势 $\hat{A}$ 时不需要混合 KL 值，更加纯粹。

### 3.4 KL 散度的无偏估计 (公式 4)

为了更稳定地计算上述公式中的 $\mathbb{D}_{KL}$，GRPO 使用了 Schulman (2020) 提出的无偏估计器：

$$\mathbb{D}_{KL}[\pi_\theta || \pi_{ref}] = \frac{\pi_{ref}(o_{i,t} | \dots)}{\pi_\theta(o_{i,t} | \dots)} - \log \frac{\pi_{ref}(o_{i,t} | \dots)}{\pi_\theta(o_{i,t} | \dots)} - 1 \quad (4)$$

这个估计器能保证 KL 值始终为正，从而稳定训练过程。

---

## 4. Figure 4 深度对比解析

让我们结合论文中的 Figure 4 来直观对比两者。

### PPO 架构 (Figure 4 上半部分)

- **输入**：问题 $q$ -> 策略模型 -> 输出 $o$。
- **计算流**：
    - 输出 $o$ 传给 Reference Model 计算 KL。
    - 输出 $o$ 传给 Reward Model 计算奖励 $r$。
    - 输出 $o$ 传给 Value Model 计算价值 $V$。
- **关键步骤**：$r$ 和 $V$ 输入到 GAE 模块，计算优势 $A$。
- **资源消耗**：需要维护 Policy (Train), Value (Train), Reference (Freeze), Reward (Freeze)。

### GRPO 架构 (Figure 4 下半部分)

- **输入**：问题 $q$ -> 策略模型 -> 一组输出 $\{o_1, o_2, ..., o_G\}$。
- **计算流**：
    - 这组输出传给 Reference Model 计算 KL。
    - 这组输出传给 Reward Model 计算一组奖励 $\{r_1, ..., r_G\}$。
- **关键步骤**：**没有 Value Model**。直接通过 Group Computation（计算均值和方差）得出优势 $\{A_1, ..., A_G\}$。
- **资源消耗**：
    - **移除了 Value Model**。
    - Reference Model 和 Reward Model 仍然存在（冻结）。
    - **计算量**：虽然 Policy 需要推理 $G$ 次，但因为是推理阶段，且不需要反向传播更新 Value Model 的庞大参数，整体显存占用和训练资源显著降低。

---

## 5. 总结

DeepSeekMath 通过引入 GRPO，针对数学推理任务做出了以下优化：

1.  **去除了 Critic (Value Model)**：解决了数学推理中 Critic 难以训练且显存占用大的问题。
2.  **Group Relative Advantage**：利用“组内对比”代替“与预期价值对比”，这在数学题（答案通常是非黑即白的，或者可以通过采样多条路径对比优劣）中非常有效。
3.  **Outcome & Process Supervision**：论文提到 GRPO 既可以用 **Outcome**（只看最终答案对错，公式3中的 $\hat{A}$ 对所有 token 一样），也可以用 **Process**（每一步给分，公式3中的 $\hat{A}$ 随步骤变化），灵活性很高。

这种方法使得 DeepSeekMath 能够利用纯强化学习（仅使用 GSM8K 和 MATH 的题目）将 7B 模型的数学能力提升到 SOTA 水平。

