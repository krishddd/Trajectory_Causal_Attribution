# Benchmarks

## Who&When step-attribution (`whowhen.py`)

Measures how accurately counterfactual step-ablation attribution localizes the
**responsible step** in a failed trajectory — the *Who&When* task
(arXiv:2505.00212), where the strongest LLM-as-judge attributor reaches only
**~14.2%** step accuracy.

```bash
python benchmarks/whowhen.py                 # default suite, 60 rollouts/step
python benchmarks/whowhen.py --rollouts 80 --adaptive
```

Representative run (synthetic suite, deterministic):

```
Who&When-style step-attribution benchmark
  cases=8  rollouts/step=60  adaptive=False

  causal attribution (this tool)  : 100.0%  (8/8)
  max-magnitude (no PoC rule)     :  12.5%  (1/8)
  last-step baseline              :  37.5%  (3/8)
  LLM-as-judge (Who&When lit.)    :  ~14.2%  (arXiv:2505.00212)
```

The three comparison rows are the point:

- **Causal attribution** uses the Point-of-Commitment rule — the *latest* step
  whose CI still excludes zero — and localizes the true culprit reliably.
- **Max-magnitude** blames the highest-|attribution| step *without* the PoC rule.
  It scores near the judge baseline because resampling an early step re-rolls the
  fatal late step too, inflating early scores (the butterfly-effect confound the
  research warns about). This row shows what the PoC rule buys.
- **Last-step** is the naive "it broke at the end" guess.

### Ground truth without the network

The public Who&When dataset is a network/licensing dependency, so this harness
ships a **synthetic generator** with known ground truth (`default_suite()`): chain
agents that are benign until a single `fail_step` commits a "BAD" action, so the
responsible step is unambiguous. Cases vary chain length and culprit position —
the axis where magnitude-based blame fails.

### Running on the real dataset

Export each real trajectory to the JSONL layout
[`agent_replay.interop`](../src/agent_replay/interop.py) reads, supply a resample
policy per step kind, build `Case`s from the imported trajectories, and call
`evaluate(cases, ...)`. The accuracy math is identical to the synthetic path.
