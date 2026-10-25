#!/usr/bin/env python3
"""Evaluate a tracking policy of either backend on an imported library in CPU MuJoCo (MorphMimic's sim2sim scene).

One nominal attempt per clip from its first frame, simulated to the clip end, scored offline with
humanoid_motion.evaluation.score_attempt; writes the same files as scripts/evaluate.py (Isaac), so
score_rollouts.py, summarize_results.py, view_tracking.py and render_video.py read every backend. Run it with
MorphMimic's interpreter (it needs their deploy/ code and onnxruntime):

    third_party/morphmimic/.venv/bin/python scripts/evaluate_mujoco.py --policy supermimic --deploy <export_onnx dir> \
        --assets <export_assets dir> --library outputs/lafan1/runtime/test.json --output outputs/eval/supermimic_test
    third_party/morphmimic/.venv/bin/python scripts/evaluate_mujoco.py --policy beyondmimic --checkpoint <.pt> ...
"""
from humanoid_motion.tracking.sim2sim import main

if __name__ == '__main__':
    main()
