---
title: agent-replay
emoji: 🔁
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# agent-replay — counterfactual step-attribution demo

Interactive demo of [agent-replay](https://github.com/krishddd/Trajectory_Causal_Attribution):
record a small stochastic multi-step agent with an injected fault, then attribute
the failure to a specific step via **counterfactual step-ablation replay** — the
culprit the tool localizes should match the fault you set.

- **Code:** <https://github.com/krishddd/Trajectory_Causal_Attribution>
- **Docs:** <https://krishddd.github.io/Trajectory_Causal_Attribution/>

This Space is a thin wrapper: it installs `agent-replay` from source
(`requirements.txt`) and defines its own demo agent in `app.py` — the library
itself ships no bundled agents.
