#!/usr/bin/env python3
"""Verify the v0.15.2 literature revision against the reviewed v0.15.1 tree.

This checks preservation, not scientific replication or hardware execution.
Use a tree ID because git-am can change commit IDs in independent histories.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = 'ca5b6f5444210d8453fcb3172d52df440635e049'
RELATED_TABLES = {f'papers/{lang}/inputs/related_work_comparison.tex'
                  for lang in ('ieee_access', 'ru')}
EDITED_TEX = RELATED_TABLES | {'papers/ieee_access/main.tex',
                             'papers/ru/main.tex', 'papers/ru/parts/a_ru.tex',
                             'papers/ru/parts/c_ru.tex'}
METHOD_SENTENCES = {
    'papers/ieee_access/main.tex': r'The uncertainty characterization in the separate CNN--KAN sensor-system study~\cite{faggi2026sensor} is specific to its measurands and setup and cannot be transferred to these USB-energy estimates. ',
    'papers/ru/parts/c_ru.tex': r'Оценка неопределённости в отдельной работе по измерительной системе CNN--KAN~\cite{faggi2026sensor} относится к её измеряемым величинам и установке и не переносится на приведённые здесь оценки USB-энергии. ',
}
ENVIRONMENTS = ('table', 'table*', 'figure', 'figure*', 'equation', 'equation*',
                'align', 'align*', 'gather', 'gather*', 'IEEEbiography')
NEW_DOIS = ('10.1109/LES.2026.3704316', '10.1016/j.neucom.2026.133774',
            '10.1016/j.sysarc.2026.103923', '10.1109/ACCESS.2026.3693881',
            '10.1109/OJIM.2026.3712887')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8')


def blocks(text, environment):
    return re.findall(r'\\begin\{' + re.escape(environment) + r'\}.*?\\end\{'
                      + re.escape(environment) + r'\}', text, re.DOTALL)


def check(condition, message):
    if not condition:
        raise SystemExit('STOP: ' + message)


def sections(text):
    """Split top-level sections; keep every other section unchanged."""
    return re.split(r'(?=\\section\{)', text)


def main():
    paths = git('ls-tree', '-r', '--name-only', BASE).splitlines()
    changes = git('diff', '--name-only', BASE, '--').splitlines()
    check(all(p.startswith('papers/') for p in changes),
          'changes outside papers/ (scientific code and evidence must be unchanged)')
    for name in changes:
        if any(x in name for x in ('/inputs/', '/figures/', '/evidence/', '/assets/')):
            check(name in RELATED_TABLES, 'changed retained asset: ' + name)
        if name.endswith('.tex'):
            check(name in EDITED_TEX, 'unexpected TeX edit: ' + name)

    counts = {env: 0 for env in ENVIRONMENTS}
    for name in paths:
        if not (name.startswith('papers/') and name.endswith('.tex')):
            continue
        previous = git('show', BASE + ':' + name)
        current = (ROOT / name).read_text(encoding='utf-8')
        if name not in RELATED_TABLES:
            for env in ENVIRONMENTS:
                old, new = blocks(previous, env), blocks(current, env)
                check(old == new, 'changed ' + env + ' block: ' + name)
                counts[env] += len(old)
        for macro in ('label', 'ref'):
            pattern = r'\\' + macro + r'\{([^}]+)\}'
            check(re.findall(pattern, previous) == re.findall(pattern, current),
                  'changed ' + macro + ' sequence: ' + name)
        if name in ('papers/ieee_access/main.tex', 'papers/ru/parts/a_ru.tex'):
            # Only the literature section may change. The bibliography and
            # biographies can be rebalanced, but their text is checked below.
            oldbody = previous.split(r'\bibliographystyle')[0].replace('0.15.1', '0.15.2')
            newbody = current.split(r'\bibliographystyle')[0]
            newbody = newbody.replace(METHOD_SENTENCES.get(name, '\0'), '')
            oldparts, newparts = sections(oldbody), sections(newbody)
            check(len(oldparts) == len(newparts), 'changed section structure: ' + name)
            for old, new in zip(oldparts, newparts):
                if old.startswith(r'\section{Related Work and Scope}') or old.startswith(r'\section{Связанные работы'):
                    continue
                check(old == new, 'changed content outside related work: ' + name)
        if name == 'papers/ru/parts/c_ru.tex':
            check(current.replace(METHOD_SENTENCES[name], '') == previous,
                  'unexpected change beyond reviewed methodology sentence: ' + name)
        if name == 'papers/ru/main.tex':
            expected = previous.replace('0.15.1', '0.15.2').replace(
                r'\fontsize{8.5}{10}\selectfont', r'\fontsize{8}{9}\selectfont', 1)
            check(current == expected, 'unexpected RU metadata/layout change')
        if name == 'papers/ieee_access/main.tex':
            oldtail = previous.split(r'\bibliographystyle', 1)[1]
            newtail = current.split(r'\bibliographystyle', 1)[1]
            check(oldtail.replace('\\newpage\n', '', 1) == newtail,
                  'unexpected EN bibliography/biography change')

    en = (ROOT / 'papers/ieee_access/references.bib').read_text(encoding='utf-8')
    ru = (ROOT / 'papers/ru/references.bib').read_text(encoding='utf-8')
    # Historical documentation entries have language-specific formatting.
    # The five new records must agree exactly across both editions.
    check(en.split('@article{kuznetsov2026esl,', 1)[1] ==
          ru.split('@article{kuznetsov2026esl,', 1)[1], 'new EN/RU references differ')
    for doi in NEW_DOIS:
        check(doi in en, 'missing new DOI: ' + doi)
    print('EDITORIAL_V0152_PASS: scientific code/evidence, experimental tables,')
    print('figures, equations, labels and non-literature body sections preserved.')
    print('Reviewed exceptions: related-work table, five journal references, one')
    print('methodology-scope sentence per language and checked final-page layout.')
    print('Baseline tree:', BASE, '| protected blocks:', counts)
    print('This check does not establish coauthor approval or new scientific results.')


if __name__ == '__main__':
    main()
