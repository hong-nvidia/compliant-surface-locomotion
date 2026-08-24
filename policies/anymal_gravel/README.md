# Bundled ANYmal gravel policy

`model.pt` is the repository's default RSL-RL checkpoint for
`scripts/view_anymal_policy.py`. Git LFS stores the binary; after cloning, run:

```bash
git lfs pull
```

## Provenance

- Training run: `2026-08-19_19-07-10_upright_reward_v1`
- Original filename: `model_interrupt.pt`
- Saved PPO iteration: 4439
- Seed: 42
- Environment steps per update: 8 environments × 96 steps
- SHA-256: `f59d6cf0ad6cdab50c20e5dbfe1ce17651494544e050f20acbb94efc302b6011`
- Size: 1,849,531 bytes

The adjacent `task_args.yaml` preserves the geometry, particle, timing, and task
settings needed for playback. The actor was initialized from Newton's pretrained
rigid-floor ANYmal-C policy and then fine-tuned on this repository's coupled MPM
gravel task.

## Known behavior

The final training log showed approximately 2.20 m net forward travel and 0.947
final upright cosine. This checkpoint walks across the gravel with good posture,
but stopping at the goal is not solved reliably: the final batch had zero task
successes and frequently terminated for overshooting. Treat this as a research
checkpoint, not a production controller.

To use a different policy, pass `--checkpoint`, use `--latest` for the newest
checkpoint under the local training log directory, or use `--pretrained` for
Newton's original rigid-floor actor.
