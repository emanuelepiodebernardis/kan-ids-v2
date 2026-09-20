#!/usr/bin/env python3
"""Build the Russian reading translation in an isolated directory; no device access."""
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    for tool in ('xelatex', 'bibtex', 'xdvipdfmx'):
        if not shutil.which(tool):
            raise SystemExit(f'Required on PATH: {tool}')
    generated = {'main.pdf', 'main.aux', 'main.out', 'main.log', 'main.xdv',
                 'main.bbl', 'main.blg', 'BUILD_RU.log', 'verification',
                 '__pycache__', 'MANIFEST.json'}
    logs = []
    try:
        with tempfile.TemporaryDirectory(prefix='ru_build_', dir=root) as folder:
            temporary = Path(folder)
            build = temporary / 'source'
            shutil.copytree(root, build, ignore=lambda directory, names:
                            [name for name in names if name in generated
                             or name.startswith('ru_build_') or name.endswith('.pyc')])

            def run(command):
                result = subprocess.run(command, cwd=build, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        encoding='utf-8', errors='replace')
                logs.append('$ ' + ' '.join(command) + '\n' + result.stdout + '\n')
                if result.returncode:
                    raise RuntimeError(f'BUILD_FAILED: {command[0]} returned {result.returncode}')

            latex = ['xelatex', '-no-pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex']
            run(latex)
            run(['bibtex', 'main'])
            for _ in range(6):
                run(latex)
                lines = (build/'main.log').read_text(encoding='utf-8', errors='replace').splitlines()
                if not any('Warning' in line and any(word in line.lower() for word in ('undefined', 'rerun')) for line in lines):
                    break
            else:
                raise RuntimeError('Cross-references did not stabilize within six final passes')
            bad = [line for line in lines if 'Overfull' in line or 'Missing character' in line or
                   ('Warning' in line and 'too large' in line.lower())]
            if bad:
                raise RuntimeError('LAYOUT_REVIEW_REQUIRED:\n' + '\n'.join(bad))
            run(['xdvipdfmx', '-o', 'main.pdf', 'main.xdv'])
            pdf_bytes = (build/'main.pdf').read_bytes()
            if not pdf_bytes.startswith(b'%PDF-') or not pdf_bytes.rstrip().endswith(b'%%EOF'):
                raise RuntimeError('PDF is incomplete: header or final EOF marker missing')
            if shutil.which('pdfinfo'):
                run(['pdfinfo', 'main.pdf'])
            for name in ('main.pdf', 'main.log', 'main.bbl'):
                (root/name).write_bytes((build/name).read_bytes())
    except RuntimeError as exc:
        raise SystemExit(f'{exc}; see {root / "BUILD_RU.log"}') from exc
    finally:
        with (root/'BUILD_RU.log').open('w', encoding='utf-8', newline='\n') as log:
            log.write(''.join(logs))
    print('RU_PDF_BUILD_PASS:', root/'main.pdf')
    print('Isolated build, stable references and PDF completeness verified; scientific and visual review are separate.')


if __name__ == '__main__':
    main()
