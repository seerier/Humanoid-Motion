# SuperMimic tracking backend

[SuperMimic](https://arxiv.org/abs/2610.28480) (Liao et al., CoRL 2026) trains a motion-tracking policy by backpropagating through a learned, differentiable world model instead of using PPO. Its authors have not released code. This repository uses the open, independent implementation in [MorphMimic](https://github.com/lzyang2000/morphmimic) (Yang et al., 2026; Apache-2.0), pinned at commit `8cd92b6`, in its single-robot G1 mode, with its code unmodified. MorphMimic runs on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp).

What this repository adds around it (`src/humanoid_motion/tracking/supermimic/`):

- `library.py`: converts the prepared LAFAN1 (or video) references into MorphMimic's motion library. It uses the same 50 Hz frames as the Isaac library of the BeyondMimic backend, and MorphMimic's own forward kinematics and finite differences on mjlab's G1 model. Every clip is checked against the independent URDF forward kinematics (largest difference on the LAFAN1 splits: 5.7e-7 m, 1.5e-6 rad). MorphMimic's name and feasibility filters are recorded but do not drop clips (8 of the 1,062 training clips would fail its feasibility filter), so both backends train on exactly the same clips.
- `evaluate.py` and `src/humanoid_motion/tracking/sim2sim.py`: play exported policies in MorphMimic's CPU MuJoCo sim2sim scene and write the same files as `scripts/evaluate.py`, so one offline scoring function covers both backends (`scripts/evaluate_mujoco.py`, `scripts/score_rollouts.py`).
- `returns.py`: SuperMimic's normalised return (the paper's benchmark definition) of saved rollouts.

## Switch list

Training uses MorphMimic's command-line flags only. Every switch for which the paper gives a value is set to the paper's value; the paper's gaps keep MorphMimic's defaults.

| Item | Paper | MorphMimic default | This repository | Flag |
|---|---|---|---|---|
| World-model MLP | plain (2048, 2048, 2048) (Table A.4) | LayerNorm after each hidden layer | plain | `--no-wm-layer-norm` |
| Critic MLP | plain (256, 128) (Table A.4) | LayerNorm after each hidden layer | plain | `--no-critic-layer-norm` |
| Critic targets | TD(λ) targets of Eq. 3, unclamped | clamped at 10 (largest possible return) | unclamped | `--no-critic-target-clip` |
| World-model-only warm phase | none: all networks update every iteration (Alg. A.1) | 500 iterations | none | `--wm-only-iters 0` |
| Actuator delay | none (Table A.3) | random 0 to 40 ms | none | `--no-actuator-delay` |
| Imagination start states | any replay state | at least H + K frames before the clip end | any replay state | `--no-avoid-clip-end` |
| Precision | TF32 | bfloat16 autocast | TF32 | `--no-bf16` |
| World-model residual scale | not given | one-step RMS of the warm-up replay | MorphMimic default | `--residual-scale step` |
| World-model frame | not given | pelvis frame plus gravity and body heights | MorphMimic default | `--wm-frame root` |
| Termination head | learned, form not given | hazard model on the two termination margins | MorphMimic default | `--termination hazard` |
| Reset-sampler return normalisation | not given | best return still reachable | MorphMimic default | `--reset-norm remaining` |
| Critic input | (ŝ, g, ξ) | also the reference anchor position in the robot frame | MorphMimic default | `--critic-anchor-pos` |
| Rollout, batch, learning-rate schedule | H = 32, k = 8, 4,096 envs, UTD 0.4, 2e-4 cosine to 0.03x over 250k of 320k iterations | same | same | defaults |

Not switchable, kept as MorphMimic implements them: position-free goals (the goal carries no robot position; only the critic sees it), analytic reward and observation heads, a running input normaliser for the policy, the torso as the anchor body for the "root" reward and termination terms (as in BeyondMimic and mjlab), and domain randomisation drawn once per environment at start-up.

## Training on LAFAN1

The LAFAN1 train split (1,062 six-second clips, 1.76 h) is the one the released BeyondMimic policy was trained on. One NVIDIA A40 runs about 0.85 iterations per second with 4,096 environments.

## Comparison protocol

Both backends are scored by `humanoid_motion.evaluation.score_attempt` on the 230 LAFAN1 test clips: one nominal attempt per clip from its first frame, failure at the first frame where a BeyondMimic termination fires (torso height or any ankle or wrist height more than 25 cm off, or torso tilt past the threshold). The SuperMimic paper's own failure definition (torso height and tilt only) is reported alongside. The offline scorer reproduces the online Isaac evaluator exactly (190 / 230 for the released BeyondMimic policy, the same first failure frame on all 230 clips).

The BeyondMimic policy is native to Isaac Lab (PhysX) and the SuperMimic policies to MuJoCo. To compare them in one engine, `scripts/evaluate_mujoco.py --policy beyondmimic` also runs the BeyondMimic policy in the same MuJoCo scene: its PD gains, torque limits, action scale, default pose and 5 ms x 4 control timing match the Isaac G1 to 1e-6, and its observation is rebuilt from the MuJoCo state exactly as the Isaac terms compute it.
