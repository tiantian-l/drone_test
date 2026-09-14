# Reproducible checkpoint evaluation

The `snapshot_rng0_v1` protocol is used by `train_eval` and
`cloud/eval_checkpoint.py`. Training must be paused during the evaluation context.
It copies current training weights into the evaluation policy, starts the policy
random counter at zero, and restores the training policy and pending sync after
evaluation (including on exceptions). Evaluation does not consume the training
action counter. This changes the random schedule relative to legacy training;
continuations are not expected to match legacy runs step for step.

New Top-K checkpoints are stored in `best_checkpoints_snapshot_rng0_v1/` and
their manifest records the protocol. Legacy `best_checkpoints/` is untouched.
Rolling `ckpt/` and agent checkpoint formats are unchanged. Old checkpoints can
be evaluated, but their filename scores were measured with the legacy protocol.
Use a copied run directory for initial continuation validation so old and new
evaluation curves are not silently interpreted as one protocol.

Sync both this repository and the changed `third_party/dreamerv3` submodule
working files to the cluster. Merely copying the Slurm script is insufficient.

To evaluate a new checkpoint, set `CHECKPOINT` to its full path when submitting
`cloud/eval_20_sparse_slurm.sh`. Keep the original saved run configuration and
evaluation environment count. Results go to unique output directories.

Local checks: `python cloud/test_evaluation_snapshot.py` tests snapshot ownership,
restoration, repeat initialization, nested-context rejection, and exception
cleanup using the actual context method with mock arrays. This is not GPU
validation. On a GPU, compare a newly saved checkpoint's per-map outcomes with
the training evaluation that selected it, then repeat in an independent process.
Use the same code, dependencies, precision, and hardware for this first check.
