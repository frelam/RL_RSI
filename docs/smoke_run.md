# Smoke Run

A smoke run is a brief correctness check before spending time on a formal run. It should expose obvious reward, data, configuration, and infrastructure problems. It does not estimate final model quality or select optimal hyperparameters.

## Offline workflow

The mock config checks the orchestration path without external services or a GPU:

```bash
PYTHONPATH=src python -m rsi validate --config configs/smoke_mock.yaml
PYTHONPATH=src python -m rsi run --config configs/smoke_mock.yaml --run-id smoke-mock
PYTHONPATH=src python -m rsi inspect runs/smoke-mock --history
```

This run uses mock inference, data generation, benchmarking, and training. It verifies that stages execute and artifacts are written. It says nothing about a real reward function, model, or trainer.

Every setting has a built-in default, so `--config` is optional: omit it to use `./configs/default.yaml` when present, or the packaged defaults otherwise. `--variable key.path=value` changes single settings without editing a file.

## Real trainer workflow

1. Validate the real config and component setup before allocating accelerators:

   ```bash
   PYTHONPATH=src python -m rsi validate --config configs/your_rl.yaml
   PYTHONPATH=src python -m rsi run --config configs/your_rl.yaml --run-id smoke-real --dry-run
   ```

2. Make a temporary smoke config with a small prompt/data cap, one short training attempt of about 5-10 optimizer steps, and at most one evaluation. Set the step limit using the configured trainer backend's native option. Keep the normal reward, filters, data strategy, and monitor rules enabled. Do not sweep learning rate or mixing ratios in this run.

3. Before training, inspect a handful of representative reward cases. Include a correct answer, an incorrect answer, an empty or malformed answer, and a format-only or repeated answer where applicable. Confirm that the parsed result and each reward component match expectations. Fix clear parser or verifier failures before training.

4. Run the short job and inspect its generated dataset manifest and `pipeline.json`. Confirm that expected sources/groups survived filtering, the selected counts are nonzero, the actual training shares are close enough to the configured target for this tiny sample, and the validation split did not consume the training examples. Small samples make ratios coarse, so treat them as a wiring check.

5. Check the first few metric records and logs for finite loss and gradient norm, a non-failing trainer exit, checkpoint output, and reasonable KL, entropy, response length, memory use, and step time where the backend reports them. Inspect a few high-reward outputs if accessible. A rising training reward alone is not evidence that the reward is correct.

## Real verl checklist

Before the first real run, confirm the pieces the generic adapter does not provide:

1. **Data.** Materialize enough local rows for `data.strategy.count`; the `examples/datasets` rows are placeholders. Set `data.build.type: verl_parquet` so `{train_path}` and `{val_path}` are parquet. Rows without an assistant response are marked `prompt_only` and survive the response-dependent filters.
2. **Reward.** Mount `src/rsi/rewards/verl_bridge.py` as `reward.custom_reward_function.path` with `compute_score` as its name. It reads each row's `extra_info.verifier` and dispatches to the built-in rule scorers or the `reasoning_rl` / `agentic_rl` adapters.
3. **Checkpoint.** Set `trainer.checkpoint_glob` (for example `global_step_*`) and, when verl writes sharded checkpoints, a `trainer.export` command that calls `verl.model_merger` into a Hugging Face directory. vLLM cannot serve an FSDP checkpoint directly.
4. **GPUs.** With vLLM and the trainer on the same node, set `train.release_inference: true`. If serving is pinned to other devices through `infer.launch.env`, leave it false.
5. **Metrics.** The command trainer reads `trainer.metrics_path`. verl does not write that JSONL by default; add a callback. Without it, training and benchmark validation still run, but metric monitors stay idle and the scaling-law projection has no data.

## Stop and adjust

Stop the smoke run for non-finite metrics, missing data groups or reward fields, empty training data, a failed checkpoint write, a clear reward shortcut, or an obvious resource/configuration error. Fix that issue and repeat the same short check.

If the run completes cleanly, record the tested config and proceed to a formal run. Change at most one item at a time when addressing a specific concern, such as learning rate, trainer resource settings, or a data-mixing target. Review `stats.train_group_ratio` in the dataset manifest before changing the target. A smoke run is too short and noisy to justify long-horizon tuning.

## Checks outside the current generic interface

The command-trainer adapter does not compare rollout and training log probabilities, inspect router choices, or run reward adversarial cases automatically. Add a brief backend-native log-probability comparison only when the backend already exposes it at low cost. Keep checkpoint-resume testing and repeated-seed comparisons for a separate integration or reproducibility check when those behaviors matter to the trainer setup.
