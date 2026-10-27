# Changelog

## v0.3.1

README video; models, results and the pipeline of v0.3 are unchanged.

- README video: a Pexels dance clip, the human video next to the policy that tracks it to the end (the v0.1 BeyondMimic policy, in Isaac Lab). It replaces the squat, whose reconstructed feet tilt toes-down in contact so that the G1 reference stands on its toes in about a third of the clip. Five dance videos were evaluated with both policies; the results are in the README and `docs/video.md`.
- `humanoid_motion.video.camera`: the camera GVHMR estimated for a clip, in the G1 reference frame (`camera.json`); `scripts/render_video.py video --camera video --source-camera` renders the robot with that camera angle and lens, where the person is in the frame.
- `scripts/render_video.py video`: `--no-reference` (leave out the retargeted reference column), `--source-label`, `--source-width`.

## v0.3

A video front end, a second tracking backend, and one evaluation protocol for both. The v0.1 policy is unchanged.

- Video front end: monocular video -> GVHMR -> SMPL-X -> GMR -> G1 reference in the same contract as LAFAN1 clips (`scripts/video_to_reference.py`, `humanoid_motion.video`, `scripts/setup_gvhmr.sh`, `docs/video.md`). The ground comes from GVHMR's own stationary-foot labels, frame by frame; a quality gate and a stage-by-stage audit (reconstruction, interface, retargeting, ground, tracking) say which stage a bad clip fails in. LAFAN1 retargeting output is byte-identical to v0.2.
- SuperMimic backend through MorphMimic's unofficial reimplementation in single-robot mode, unmodified (`humanoid_motion.tracking.supermimic`, `scripts/setup_supermimic.sh`, `docs/supermimic.md`): motion-library conversion from the same 50 Hz frames as the Isaac library with a URDF FK check, training with MorphMimic's flags, an exporter adapter for motion-only checkpoints, and SuperMimic's normalised return. MorphMimic is pinned at 2e35049, which has the same code as 8cd92b6, the commit the policy was trained and evaluated at (a later upstream force push removed 8cd92b6 from its main branch).
- One offline scorer for every backend (`humanoid_motion.evaluation.score_attempt`, `scripts/score_rollouts.py`), with the BeyondMimic and the SuperMimic failure criteria. It reproduces the online Isaac evaluator exactly: 190 / 230, the same first failure frame on every clip.
- MuJoCo evaluation for both backends (`scripts/evaluate_mujoco.py`, `humanoid_motion.tracking.sim2sim`). The v0.1 BeyondMimic policy runs in MuJoCo too: 190 / 230 there, as in Isaac Lab, with the same verdict on 226 clips; every attempt in both simulators starts from the reference's first frame.
- `scripts/render_video.py`: `compare` (one column per policy) and `video` (source video, retargeted reference, policies) layouts.
- Results on the 230 LAFAN1 test clips, one scorer: a SuperMimic policy trained here on the same LAFAN1 train split (MorphMimic's defaults, last checkpoint after 6.4 h on one A40) completes 195 against 190 for the v0.1 BeyondMimic policy (22 h), 224 against 216 under SuperMimic's torso-only criterion, with lower pose and drift errors. With 20 ms of actuator delay the BeyondMimic policy drops to 161 and the SuperMimic policy to 193. MorphMimic's cross-embodiment `unified_v2` policy is evaluated zero-shot as a reference in `docs/supermimic.md` (198), not in the main comparison. A first SuperMimic run with the paper's switch values diverged near iteration 3,000.
- Release asset: the SuperMimic policy trained here (`supermimic_lafan1_24000.zip`: MorphMimic's `policy.onnx` and `g1.json`), for `scripts/evaluate_mujoco.py --policy supermimic --deploy`.
- `render_video.py` shadow box widened so that NVIDIA EGL renders no speckles on the far floor; `--delay-ms` in the MuJoCo evaluation.

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
