#!/usr/bin/env python3
"""Fetch pinned G1 URDF for audit, optionally including runtime G1 meshes."""
import argparse
from pathlib import Path
import hashlib
import io
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'third_party/whole_body_tracking'
URL = 'https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz'
SHA = 'b514bc9ddd1039c29a0e6feea9f57f1503f6657d07d97a4ef8a7b11fbebe6674'
MEMBER = 'unitree_description/urdf/g1/main.urdf'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true', help='Also stage G1-only URDF/meshes in tracker assets')
    args = parser.parse_args()
    archive = DEST / 'unitree_description.tar.gz'
    raw = archive.read_bytes() if archive.exists() else urllib.request.urlopen(URL, timeout=120).read()
    if hashlib.sha256(raw).hexdigest() != SHA:
        raise ValueError('Robot archive checksum changed; inspect before adopting it')
    DEST.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        archive.write_bytes(raw)
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tar:
        content = tar.extractfile(MEMBER).read()
        if args.runtime:
            asset_root = DEST / 'source/whole_body_tracking/whole_body_tracking/assets'
            selected = [m for m in tar.getmembers() if m.name.startswith(
                ('unitree_description/urdf/g1/', 'unitree_description/meshes/g1/'))
                or m.name == 'unitree_description/package.xml']
            for member in selected:
                if member.isdir():
                    continue
                if not member.isfile() or '..' in Path(member.name).parts:
                    raise ValueError(f'Unsafe asset member: {member.name}')
                target = asset_root / member.name
                if not target.resolve().is_relative_to(asset_root.resolve()):
                    raise ValueError('Asset path escapes project asset tree')
                payload = tar.extractfile(member).read()
                if target.exists() and target.read_bytes() != payload:
                    raise ValueError(f'Existing asset changed: {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
            print(f'G1 runtime assets staged under {asset_root.relative_to(ROOT)}')
    dest = DEST / 'audited_asset' / MEMBER
    if dest.exists() and dest.read_bytes() != content:
        raise ValueError('Existing URDF changed; refusing overwrite')
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    print(f'Verified robot archive; extracted {dest.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
