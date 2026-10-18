"""Simulator-independent library contracts, atomic artifacts and sampling utilities."""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from humanoid_motion.contracts import sha256, validate_tracker
from humanoid_motion.evaluation import validate_transfer_contract


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def bounded_probabilities(failure_rates: np.ndarray, mixed: bool) -> np.ndarray:
    """50% uniform plus capped difficulty, with redistribution preserving the simplex."""
    rates = np.asarray(failure_rates, dtype=float)
    if rates.ndim != 1 or not len(rates) or not np.isfinite(rates).all() or (rates < 0).any():
        raise ValueError('Finite nonnegative clip failure rates required')
    n = len(rates)
    if not mixed:
        return np.full(n, 1 / n)
    weights = rates + .01
    # Difficulty component cap 3/N gives final .5/N <= p <= 2/N.
    low, high = 0., 3. / (n * weights.min())
    for _ in range(60):
        middle = (low + high) / 2
        if np.minimum(weights * middle, 3 / n).sum() < 1:
            low = middle
        else:
            high = middle
    difficulty = np.minimum(weights * high, 3 / n)
    difficulty /= difficulty.sum()
    return .5 / n + .5 * difficulty


def flat_indices(lengths: np.ndarray, clips: np.ndarray, phases: np.ndarray) -> np.ndarray:
    lengths = np.asarray(lengths)
    clips, phases = np.asarray(clips), np.asarray(phases)
    if lengths.ndim != 1 or (lengths < 3).any() or clips.shape != phases.shape:
        raise ValueError('Invalid library dimensions')
    if (clips < 0).any() or (clips >= len(lengths)).any() or (phases < 0).any() or (phases >= lengths[clips]).any():
        raise ValueError('Clip or phase out of range')
    return np.r_[0, np.cumsum(lengths)[:-1]][clips] + phases


def load_library(path: Path, root: Path) -> tuple[dict, list[dict], dict]:
    manifest = json.loads(path.read_text())
    if manifest['schema'] != 'humanoid_motion.library.isaac.v1':
        raise ValueError('Expected audited Isaac library')
    arrays, ids, contract = [], set(), None
    for row in manifest['clips']:
        if row['id'] in ids:
            raise ValueError('Duplicate clip ID')
        ids.add(row['id'])
        source = (root / row['motion']).resolve()
        if not source.is_relative_to(root.resolve()) or sha256(source) != row['motion_sha256']:
            raise ValueError('Unsafe path or motion hash mismatch')
        sidecar_path = source.with_suffix('.manifest.json')
        if sha256(sidecar_path) != row['runtime_manifest_sha256']:
            raise ValueError('Motion contract hash mismatch')
        sidecar = json.loads(sidecar_path.read_text())
        if not sidecar['audit_passed'] or sidecar['motion_sha256'] != row['motion_sha256']:
            raise ValueError('Runtime audit missing')
        if contract is not None:
            validate_transfer_contract(contract, sidecar)
        else:
            contract = sidecar
        with np.load(source, allow_pickle=False) as archive:
            data = dict(archive)
        validate_tracker(data, sidecar, contract['joint_names'], contract['body_names'], .02)
        if len(data['joint_pos']) != row['frames']:
            raise ValueError('Incorrect clip length')
        arrays.append(data)
    if not arrays:
        raise ValueError('Empty library')
    return manifest, arrays, contract

