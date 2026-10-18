# Changelog

## v0.1

First release: LAFAN1 to Unitree G1, GMR retargeting backend and BeyondMimic tracking backend.

- Retargeting: whole-recording GMR retargeting, per-recording ground offset, 6 s windows, symbolic audit, stratified split; optional IK floor constraint.
- Reference preparation: 50 Hz resampling, independent URDF kinematics, audited Isaac kinematic import.
- Training: multi-motion PPO on the BeyondMimic environment with failure-adaptive sampling, future-window observations and exact resume.
- Evaluation: fixed-start evaluator, browser replay, MuJoCo video rendering, result tables.
- Released policy: 190 / 230 held-out test clips completed.
