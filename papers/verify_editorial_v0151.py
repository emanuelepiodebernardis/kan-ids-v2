#!/usr/bin/env python3
"""Check that the v0.15.1 editorial revision preserves the v0.15.0 evidence.

Requires the delivered Git history. This is a preservation check, not a new
scientific replication, model fit or device measurement.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# git-am preserves the exact reviewed tree, but not necessarily commit IDs.
# Use the v0.15.0 tree so this also works in a student's independent history.
BASE = 'a763684d1be5ee68b678b935400466149d603ded'
PLACEMENT_ONLY = 'papers/ieee_access/inputs/factorial_table.tex'
ENVIRONMENTS = ('table', 'table*', 'figure', 'figure*', 'equation', 'equation*',
                'align', 'align*', 'gather', 'gather*')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8')


def blocks(text, environment):
    pattern = (r'\\begin\{' + re.escape(environment) + r'\}.*?\\end\{'
               + re.escape(environment) + r'\}')
    return re.findall(pattern, text, flags=re.DOTALL)


def main():
    baseline_paths = git('ls-tree', '-r', '--name-only', BASE).splitlines()
    changes = git('diff', '--name-only', BASE, '--').splitlines()
    forbidden = [name for name in changes if not name.startswith('papers/')]
    if forbidden:
        raise SystemExit('STOP: non-editorial paths changed: ' + ', '.join(forbidden))

    protected = ('/inputs/', '/figures/', '/evidence/', '/assets/')
    immutable_paper_changes = [name for name in changes
                               if any(x in name for x in protected)
                               and name != PLACEMENT_ONLY]
    if immutable_paper_changes:
        raise SystemExit('STOP: retained paper assets changed: ' + ', '.join(immutable_paper_changes))

    counts = {env: 0 for env in ENVIRONMENTS}
    tex_files = [name for name in baseline_paths
                 if name.startswith('papers/') and name.endswith('.tex')]
    for name in tex_files:
        previous = git('show', BASE + ':' + name)
        current = (ROOT / name).read_text(encoding='utf-8')
        if name == PLACEMENT_ONLY:
            # Sole typesetting exception: permit this table to float. Its entire
            # body, caption, numbers and every other byte must remain unchanged.
            permitted = previous.replace(r'\begin{table}[H]', r'\begin{table}[t]', 1)
            if current not in (previous, permitted):
                raise SystemExit('STOP: change exceeds table placement in ' + name)
            current = previous
        for environment in ENVIRONMENTS:
            old, new = blocks(previous, environment), blocks(current, environment)
            if old != new:
                raise SystemExit('STOP: changed ' + environment + ' block in ' + name)
            counts[environment] += len(new)
        for macro in ('label', 'ref', 'cite'):
            pattern = r'\\' + macro + r'\{([^}]+)\}'
            if re.findall(pattern, previous) != re.findall(pattern, current):
                raise SystemExit('STOP: changed ' + macro + ' sequence in ' + name)

    print('EDITORIAL_V0151_PASS: all changes are under papers/; retained input tables,')
    print('figures, evidence, equations, labels and citation sequences match v0.15.0,')
    print('apart from the explicitly checked factorial-table placement [H] -> [t].')
    print('Baseline:', BASE, '| checked TeX files:', len(tex_files), '| blocks:', counts)
    print('This check does not establish new scientific validity or hardware execution.')


if __name__ == '__main__':
    main()
