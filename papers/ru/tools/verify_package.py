#!/usr/bin/env python3
"""Verify the exact payload of this extracted checkpoint or its ZIP; no device access."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def verify(target):
    archive = None
    if target.is_file():
        archive = zipfile.ZipFile(target)
        infos = archive.infolist()
        names = [i.filename for i in infos if not i.is_dir()]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP paths')
        if archive.testzip() is not None:
            raise ValueError('ZIP CRC failure')
        read = archive.read
    else:
        names = [p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()]
        read = lambda name: (target / name).read_bytes()
    try:
        manifest = json.loads(read('MANIFEST.json'))
        files = manifest['files']
        expected = [f['path'] for f in files]
        if len(expected) != len(set(expected)):
            raise ValueError('Duplicate manifest paths')
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:
                raise ValueError('Unsafe path: ' + name)
        if set(names) != set(expected) | {'MANIFEST.json'}:
            raise ValueError('File set mismatch; missing=' + str(sorted(set(expected)-set(names))) +
                             '; extra=' + str(sorted(set(names)-set(expected)-{'MANIFEST.json'})))
        for item in files:
            data = read(item['path'])
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('Payload mismatch: ' + item['path'])
        return {'status': 'CHECKPOINT_MANIFEST_PASS', 'version': manifest['version'],
                'verified_payload_files': len(files),
                'note': 'Integrity verification is not scientific replication or hardware execution.'}
    finally:
        if archive:
            archive.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', nargs='?', type=Path, default=Path('.'))
    args = parser.parse_args()
    print(json.dumps(verify(args.target), indent=2))
