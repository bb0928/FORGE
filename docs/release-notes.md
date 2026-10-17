# Release preparation

## Preserved

- August 1 milestone working-tree snapshot, including the uncommitted payload XML,
  reward and runner changes. The underlying Git commit by itself is insufficient
  to recover this snapshot.
- Reward functions, runner, PPO, network weights layout, observation contract,
  symmetry, command generation and FK implementation.
- Training command mixture, pose guard, curriculum, symmetry and StdL2 settings.
- Only the robot asset variant used by the milestone and its referenced meshes.

## Changed for portability

- Removed the private W&B endpoint and mounted-storage defaults. Logs are local;
  the release launcher selects TensorBoard and disables W&B.
- Added dependency-free train/replay launchers and an explicit milestone profile.
  Launchers do not inherit stale training or distributed-process settings.
- Removed the unused frozen-base import and asset alternatives from the training
  entry point; full-body policy behavior is unchanged.
- Replaced misleading walk-only/V34 documentation with the actual mixed-command
  milestone behavior and current observation dimensions.
- Viewer curriculum now starts from checkpoint state within the 0.6 cap, rather
  than forcing 1.0. Checkpoint switching updates that control too.
- Removed duplicate/unused overlay code; added finite headless replay, checkpoint
  existence checks, non-finite checks, and localhost binding.
- Added licenses, dependency pins, provenance hashes and validation documentation.

## Excluded

Internal scheduler jobs and launch scripts, workflow identifiers, experiments,
old/broken residual evaluation tools, obsolete global PPO patch installers, backup
robot models, unused action/observer/force-control modules, old README text, logs,
screen captures, checkpoints, caches, credentials and previous Git history.

## Release boundary

This is a cleaned source release candidate. Original `model_9999.pt` weights and
the original complete package freeze were not present in the selected local
snapshot. The supplied dependency baseline is reconstructed from source/API and
asset evidence. A passing CPU check is not a CUDA training reproduction or a
confirmation that the historical checkpoint achieves its recorded performance.
