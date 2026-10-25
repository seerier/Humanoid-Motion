# SuperMimic tracking backend

[SuperMimic](https://arxiv.org/abs/2610.28480) (Liao et al., CoRL 2026) trains a motion-tracking policy by backpropagating through a learned, differentiable world model instead of using PPO. Its authors have not released code. This repository uses the open, independent implementation in [MorphMimic](https://github.com/lzyang2000/morphmimic) (Yang et al., 2026; Apache-2.0), pinned at commit `8cd92b6`, in its single-robot G1 mode, with its code unmodified. MorphMimic runs on [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp).

What this repository adds around it (`src/humanoid_motion/tracking/supermimic/`):

- `library.py`: converts the prepared LAFAN1 (or video) references into MorphMimic's motion library. It uses the same 50 Hz frames as the Isaac library of the BeyondMimic backend, and MorphMimic's own forward kinematics and finite differences on mjlab's G1 model. Every clip is checked against the independent URDF forward kinematics (largest difference on the LAFAN1 splits: 5.7e-7 m, 1.5e-6 rad). MorphMimic's name and feasibility filters are recorded but do not drop clips (8 of the 1,062 training clips would fail its feasibility filter), so both backends train on exactly the same clips.
- `evaluate.py` and `src/humanoid_motion/tracking/sim2sim.py`: play exported policies in MorphMimic's CPU MuJoCo sim2sim scene and write the same files as `scripts/evaluate.py`, so one offline scoring function covers both backends (`scripts/evaluate_mujoco.py`, `scripts/score_rollouts.py`).
- `returns.py`: SuperMimic's normalised return (the paper's benchmark definition) of saved rollouts.

## Switch list

Training uses MorphMimic's command-line flags only. A first LAFAN1 run set every switch for which the paper gives a value to the paper's value (column "Run 1") and kept MorphMimic's defaults for the paper's gaps. It diverged near iteration 3,000: the critic loss rose about fiftyfold within 100 iterations, the value estimate approached its upper bound (10) and the episode length and return fell; it had no checkpoint from before the divergence. The policy reported here (column "Run 2") is trained with MorphMimic's own defaults throughout, the recipe behind its released policies, rather than a mix of the two.

| Item | Paper | MorphMimic default | Run 1 (diverged) | Run 2 (reported) | Flag for the paper value |
|---|---|---|---|---|---|
| World-model MLP | plain (2048, 2048, 2048) (Table A.4) | LayerNorm after each hidden layer | plain | LayerNorm | `--no-wm-layer-norm` |
| Critic MLP | plain (256, 128) (Table A.4) | LayerNorm after each hidden layer | plain | LayerNorm | `--no-critic-layer-norm` |
| Critic targets | TD(λ) targets of Eq. 3, unclamped | clamped at 10 (largest possible return) | unclamped | clamped | `--no-critic-target-clip` |
| World-model-only warm phase | none: all networks update every iteration (Alg. A.1) | 500 iterations | none | 500 iterations | `--wm-only-iters 0` |
| Actuator delay | none (Table A.3) | random 0 to 40 ms | none | 0 to 40 ms | `--no-actuator-delay` |
| Imagination start states | any replay state | at least H + K frames before the clip end | any replay state | away from clip ends | `--no-avoid-clip-end` |
| Precision | TF32 | bfloat16 autocast | TF32 | bfloat16 | `--no-bf16` |
| World-model residual scale | not given | one-step RMS of the warm-up replay | default | default | |
| World-model frame | not given | pelvis frame plus gravity and body heights | default | default | |
| Termination head | learned, form not given | hazard model on the two termination margins | default | default | |
| Reset-sampler return normalisation | not given | best return still reachable | default | default | |
| Critic input | (ŝ, g, ξ) | also the reference anchor position in the robot frame | default | default | |
| Rollout, batch, learning-rate schedule | H = 32, k = 8, 4,096 envs, UTD 0.4, 2e-4 cosine to 0.03x over 250k of 320k iterations | same | same | same | |

Not switchable, kept as MorphMimic implements them: position-free goals (the goal carries no robot position; only the critic sees it), analytic reward and observation heads, a running input normaliser for the policy, the torso as the anchor body for the "root" reward and termination terms (as in BeyondMimic and mjlab), and domain randomisation drawn once per environment at start-up.

## Training on LAFAN1

The LAFAN1 train split (1,062 six-second clips, 1.76 h) is the one the released BeyondMimic policy was trained on. With the paper's TF32 setting one NVIDIA A40 ran about 0.86 iterations per second with 4,096 environments.

## Comparison protocol

Both backends are scored by `humanoid_motion.evaluation.score_attempt` on the 230 LAFAN1 test clips: one nominal attempt per clip from its first frame, failure at the first frame where a BeyondMimic termination fires (torso height or any ankle or wrist height more than 25 cm off, or torso tilt past the threshold). The SuperMimic paper's own failure definition (torso height and tilt only) is reported alongside. The offline scorer reproduces the online Isaac evaluator exactly (190 / 230 for the released BeyondMimic policy, the same first failure frame on all 230 clips).

The BeyondMimic policy is native to Isaac Lab (PhysX) and the SuperMimic policies to MuJoCo. To compare them in one engine, `scripts/evaluate_mujoco.py --policy beyondmimic` also runs the BeyondMimic policy in the same MuJoCo scene: its PD gains, torque limits, action scale, default pose and 5 ms x 4 control timing match the Isaac G1 to 1e-6, and its observation is rebuilt from the MuJoCo state exactly as the Isaac terms compute it.
