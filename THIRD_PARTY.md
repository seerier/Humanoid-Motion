# Third-party components, data and licenses

This repository's own code is MIT-licensed (see `LICENSE`). It depends on, but does not redistribute, the components below. `scripts/setup_cpu.sh` and `scripts/setup_isaac.sh` fetch them at the commits pinned in `configs/upstream_pins.json`.

| Component | Source | License | How it is used here |
|---|---|---|---|
| GMR (General Motion Retargeting) | [YanjieZe/GMR](https://github.com/YanjieZe/GMR) | MIT | Retargeting backend. Unmodified pinned checkout in `third_party/GMR`; imported from there. Its supplied robot models and meshes are used in place. |
| BeyondMimic tracking code | [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | MIT | Tracking backend: G1 environment, rewards, failure terms, PPO settings. Unmodified pinned checkout; this repository subclasses its motion command and replaces its observations. |
| Isaac Lab v2.1.0, Isaac Sim 4.5.0 | [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab), NVIDIA | BSD-3-Clause (Isaac Lab); NVIDIA license terms (Isaac Sim) | Simulator for import, training and evaluation. Installed by `scripts/setup_isaac.sh`. |
| rsl_rl | [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) | BSD-3-Clause | PPO implementation, via Isaac Lab's `isaaclab_rl`. |
| Unitree G1 description (URDF and meshes) | Archive linked from the pinned BeyondMimic README | No license file found in the archive | Downloaded and checksum-verified by `scripts/fetch_robot_asset.py`; not redistributed. |
| LAFAN1 motion capture | [ubisoft/ubisoft-laforge-animation-dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | CC BY-NC-ND 4.0 | Source motion. Downloaded and checksum-verified by `scripts/setup_cpu.sh`; BVH files and retargeted clips are not redistributed. |
| Python packages | PyPI, see `requirements/` | Their respective licenses | Exact versions pinned in `requirements/cpu.lock.txt` and `requirements/isaac-linux.lock.txt`. |

## Released policy weights and videos

The released policy checkpoint and the demo videos are derived from training on and rendering LAFAN1 motions. They are provided for **non-commercial research use**, with attribution to Ubisoft La Forge for the underlying motion data. The MIT license of this repository's code does not extend to them.

## Citations

- LAFAN1: Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, Christopher Pal. *Robust Motion In-betweening.* ACM Transactions on Graphics (SIGGRAPH), 2020.
- GMR: João Pedro Araújo, Yanjie Ze, Pei Xu et al. *Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking.* ICRA 2026. [arXiv:2510.02252](https://arxiv.org/abs/2510.02252)
- BeyondMimic: Qiayuan Liao, Takara E. Truong, Xiaoyu Huang et al. *BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion.* 2025 (preprint). [arXiv:2508.08241](https://arxiv.org/abs/2508.08241)
