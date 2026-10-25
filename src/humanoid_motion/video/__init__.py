"""Video front end: monocular video -> GVHMR (official demo, unmodified) -> SMPL-X -> GMR -> Unitree G1 reference.

gvhmr_adapter turns GVHMR's `hmr4d_results.pt` into an AMASS-style SMPL-X npz (z up, grounded) that
humanoid_motion.retarget.sources reads; retarget_config writes the GMR config; gate scores the result (routing only);
reconcile, stage_audit and fk_check are verification tools. The end-to-end entry point is scripts/video_to_reference.py,
which hands the result to humanoid_motion.retarget.reference.prepare, so the tracking side sees the same reference
contract as LAFAN1. GVHMR and the SMPL / SMPL-X body models are not distributed here; see docs/video.md.
"""
