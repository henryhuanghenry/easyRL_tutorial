# Easy-RL 高密度教程

把 [Easy-RL](https://datawhalechina.github.io/easy-rl/) 章节 + 经典 RL/RLHF 论文整理成 **Obsidian 高密度教程**，强调推导链完整、工程坑讲清、一图总览。

## 教程清单

按学习顺序：

| # | 教程 | 主题 |
|---|---|---|
| 1 | [强化学习基础教程](强化学习基础教程.md) | RL 三要素 / MDP/POMDP / 探索利用 / 多臂赌博机 |
| 2 | [马尔可夫决策过程教程](马尔可夫决策过程教程.md) | 贝尔曼方程、策略迭代、价值迭代 |
| 3 | [表格型方法教程](表格型方法教程.md) | 蒙特卡洛、时序差分、Sarsa、Q-learning |
| 4 | [策略梯度教程](策略梯度教程.md) | REINFORCE、Log-Derivative Trick、降方差 |
| 5 | [PPO教程](PPO教程.md) | 重要性采样、TRPO → PPO-Clip / PPO-Penalty |
| 6 | [DQN教程](DQN教程.md) | Q 网络、Replay Buffer、Target Network |
| 7 | [DQN进阶技巧教程](DQN进阶技巧教程.md) | Double / Dueling / PER / Noisy / Distributional / Rainbow |
| 8 | [连续动作DQN教程](连续动作DQN教程.md) | NAF / DDPG / TD3 / QT-Opt |
| 9 | [Actor-Critic教程](Actor-Critic教程.md) | A2C / A3C / GAE / DDPG |
| 10 | [DPO教程](DPO教程.md) | 直接偏好优化（arXiv 2305.18290） |
| 11 | [GRPO教程](GRPO教程.md) | 组相对策略优化（DeepSeekMath / R1） |

写作规范见 [_总结流程与要求.md](_总结流程与要求.md)。

## 渲染成 PDF（飞机离线读）

仓库已含 HTML 产物（`html_export/`）。从 HTML 出 PDF：

```bash
# 1) 生成 HTML（如果改了 md）
uv pip install --system markdown-it-py mdit-py-plugins pyyaml
python3 tools/render_md_to_html.py --all --out-dir html_export

# 2) HTML → PDF（Chrome headless）
mkdir -p pdf_export
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for html in html_export/*.html; do
  name=$(basename "$html" .html)
  "$CHROME" --headless=new --disable-gpu --no-pdf-header-footer \
    --virtual-time-budget=15000 --run-all-compositor-stages-before-draw \
    --print-to-pdf="pdf_export/${name}.pdf" --print-to-pdf-no-header \
    "file://$(pwd)/$html"
done
```

CSS 已针对 A4 做打印优化：本地 SVG 90% 宽，外部 PNG 65% 宽居中，callout/表格/代码块加 `page-break-inside: avoid`。

`pdf_export/` 不进仓库（38M），跑一次 ~30 秒。

## 风格

- 每篇含：abstract callout、形式化、分 Step 推导、直观解读、实现技巧、Cheat Sheet、graphviz 一图总览、关联笔记
- 公式块用 `\boxed{}` 高亮关键结论，`\tag{}` 保留原文公式编号
- Obsidian callout 类型：abstract / note / tip / warning / danger / success / info / example / summary / checklist
- Graphviz 配色约定：起点浅蓝、关键 trick 浅黄、改进版浅绿、最终 SOTA 浅红

## 目录

```
.
├── *教程.md                  # 11 篇教程
├── _总结流程与要求.md         # 写作规范
├── tools/
│   └── render_md_to_html.py # md → html 渲染脚本（A4 打印优化）
└── html_export/             # 渲染产物
    ├── _assets/style.css
    └── *.html
```
