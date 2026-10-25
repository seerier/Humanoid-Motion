# Third-party components, data and licenses

This repository's own code is MIT-licensed (see `LICENSE`). It depends on, but does not redistribute, the components below. `scripts/setup_cpu.sh`, `scripts/setup_isaac.sh` and the optional `scripts/setup_gvhmr.sh` and `scripts/setup_supermimic.sh` fetch them at the commits pinned in `configs/upstream_pins.json`.

| Component | Source | License | How it is used here |
|---|---|---|---|
| GMR (General Motion Retargeting) | [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | MIT | Retargeting backend. Unmodified pinned checkout in `third_party/GMR`; imported from there. Its supplied robot models and meshes are used in place. |
| BeyondMimic tracking code | [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | MIT | Tracking backend: G1 environment, rewards, failure terms, PPO settings. Unmodified pinned checkout; this repository subclasses its motion command and replaces its observations. |
| Isaac Lab v2.1.0, Isaac Sim 4.5.0 | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab), NVIDIA | BSD-3-Clause (Isaac Lab); NVIDIA license terms (Isaac Sim) | Simulator for import, training and evaluation. Installed by `scripts/setup_isaac.sh`. |
| rsl_rl | [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | BSD-3-Clause | PPO implementation, via Isaac Lab's `isaaclab_rl`. |
| Unitree G1 description (URDF and meshes) | Archive linked from the pinned BeyondMimic README | No license file found in the archive | Downloaded and checksum-verified by `scripts/fetch_robot_asset.py`; not redistributed. |
| LAFAN1 motion capture | [ubisoft/ubisoft-laforge-animation-dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | CC BY-NC-ND 4.0 | Source motion. Downloaded and checksum-verified by `scripts/setup_cpu.sh`; BVH files and retargeted clips are not redistributed. |
| GVHMR (optional video front end) | [zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) | Educational, research and non-profit use only; derivatives must be open source; no commercial use (its `LICENSE`) | Runs the official demo from a pinned checkout in its own environment; installed by `scripts/setup_gvhmr.sh`, which downloads its weights and checks them by SHA-256. Not vendored, weights not redistributed. |
| SMPL-X and SMPL body models; `smplx` package | [MPI for Intelligent Systems](https://smpl-x.is.tue.mpg.de/) | Non-commercial scientific research licences | The user registers and downloads the models; the `smplx` package is an optional extra (`video`), not in the default lock. Not redistributed. |
| MorphMimic (optional SuperMimic backend) | [lzyang2000/morphmimic](https://github.com/lzyang2000/morphmimic) | Apache-2.0 | Independent open implementation of SuperMimic. Unmodified pinned checkout in `third_party/morphmimic`, with its own locked environment (`scripts/setup_supermimic.sh`); this repository imports its training, export and sim2sim code. |
| MorphMimic released `unified_v2` policy | MorphMimic live demo, `https://lzyang2000.github.io/morphmimic/demo/live/policy.onnx` | Not stated separately from the code | Evaluated here (zero-shot results and videos); not redistributed. |
| mjlab, MuJoCo, MuJoCo Warp, NVIDIA Warp | [mujocolab/mjlab](https://github.com/mujocolab/mjlab), [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco), NVIDIA | Apache-2.0 | Simulator stack of the SuperMimic backend and of the MuJoCo evaluation, through MorphMimic's environment. onnxruntime (MIT) runs exported policies. |
| Video clips used for the video results | [Pexels](https://www.pexels.com/license/) (squat clip 8837118); [CMU Panoptic Studio](http://domedb.perception.cs.cmu.edu/) (dataset 171204_pose3) | Pexels licence (free use and modification, attribution not required); Panoptic: research use | The Pexels clip appears in the README video. The two Panoptic windows are used for evaluation only; their frames are not redistributed. |
| Python packages | PyPI, see `requirements/` | Their respective licenses | Exact versions pinned in `requirements/cpu.lock.txt` and `requirements/isaac-linux.lock.txt`; MorphMimic's environment by its `uv.lock`. |

## Released policy weights and videos

The released policy checkpoint and the demo videos are derived from training on and rendering LAFAN1 motions. They are provided for **non-commercial research use**, with attribution to Ubisoft La Forge for the underlying motion data. The MIT license of this repository's code does not extend to them.

## Citations

- LAFAN1: Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, Christopher Pal. *Robust Motion In-betweening.* ACM Transactions on Graphics (SIGGRAPH), 2020.
- GMR: João Pedro Araújo, Yanjie Ze, Pei Xu et al. *Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking.* ICRA 2026. [arXiv:2510.02252](https://arxiv.org/abs/2510.02252)
- BeyondMimic: Qiayuan Liao, Takara E. Truong, Xiaoyu Huang et al. *BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion.* 2025 (preprint). [arXiv:2508.08241](https://arxiv.org/abs/2508.08241)
- SuperMimic: Liao et al. *SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model.* CoRL 2026. [arXiv:2610.28480](https://arxiv.org/abs/2610.28480)
- MorphMimic: Lizhi Yang, Fengze Xie, Yuyang Wang, Lu Gan, Yisong Yue, Aaron D. Ames. *MorphMimic: Morphology-Conditioned World Model and Policy Sharing Across Humanoids.* Technical report, 2026. [Project page](https://lzyang2000.github.io/morphmimic/)
- GVHMR: Zehong Shen, Huaijin Pi, Yan Xia et al. *World-Grounded Human Motion Recovery via Gravity-View Coordinates.* SIGGRAPH Asia 2024. [arXiv:2409.06662](https://arxiv.org/abs/2409.06662)
- SMPL-X: Georgios Pavlakos, Vasileios Choutas, Nima Ghorbani et al. *Expressive Body Capture: 3D Hands, Face, and Body from a Single Image.* CVPR 2019.
