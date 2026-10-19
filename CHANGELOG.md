# Changelog

## v0.2

Demo media and video rendering; the policy and its results are unchanged.

- README teaser: one policy on six held-out test clips in a 3 × 2 grid. Pipeline figure: human motion capture, retargeted G1 reference and physics-based tracking side by side. Both are animated WebP at 50 fps.
- `scripts/render_video.py`: the G1 in its own materials on a checker floor, a key light with shadows that follows the robot, per-panel labels and white gutters; writes an animated WebP or GIF next to the MP4. Defaults are now three 400 × 300 panels per row. The reference ghost is opt-in (`--ghost`) and meant for analysis.
- Cinematic camera by default: a 3/4 front view that turns with the body's smoothed heading, a 28° lens framed on the body's size from near eye level, lead room in the direction of travel and a ±8° swing over the clip, at 50 fps (the rate the rollouts are recorded at). `--camera fixed`, `--follow spring`, `--orbit` and `--blur` (motion blur) are options.

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
