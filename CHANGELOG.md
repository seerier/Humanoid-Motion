# Changelog

## v0.1.1

Fixes for the CPU environment on Linux; models and results are unchanged.

- `scripts/setup_cpu.sh` installs the CPU-only PyTorch wheel on Linux. Before, it installed the CUDA build without its NVIDIA libraries, so importing GMR failed.
- The G1 description download verifies TLS with certifi's CA bundle, so it also works on RHEL-family hosts such as Rocky Linux.

## v0.1

First release: LAFAN1 to Unitree G1, GMR retargeting backend and BeyondMimic tracking backend.

- Retargeting: whole-recording GMR retargeting, per-recording ground offset, 6 s windows, symbolic audit, stratified split; optional IK floor constraint.
- Reference preparation: 50 Hz resampling, independent URDF kinematics, audited Isaac kinematic import.
- Training: multi-motion PPO on the BeyondMimic environment with failure-adaptive sampling, future-window observations and exact resume.
- Evaluation: fixed-start evaluator, browser replay, MuJoCo video rendering, result tables.
- Released policy: 190 / 230 held-out test clips completed.
