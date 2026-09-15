#!/usr/bin/env python3
"""Build the IEEE Access working manuscript in an isolated temporary directory."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-dir', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.paper_dir.resolve()
    for tool in ('pdflatex', 'bibtex'):
        if shutil.which(tool) is None:
            parser.error(f'{tool} is required on PATH (TeX Live or MiKTeX).')
    logs = []
    generated = {'main.pdf', 'main.aux', 'main.out', 'main.log', 'main.bbl', 'main.blg',
                 'BUILD.log', 'verification', '__pycache__', 'MANIFEST.json'}
    try:
        with tempfile.TemporaryDirectory(prefix='kanids-paper-') as folder:
            build = Path(folder) / 'source'
            shutil.copytree(root, build, ignore=lambda directory, names:
                            [name for name in names if name in generated or name.endswith('.pyc')])

            def run(command):
                result = subprocess.run(command, cwd=build, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, text=True,
                                        encoding='utf-8', errors='replace', check=False)
                logs.append('$ ' + ' '.join(command) + '\n' + result.stdout + '\n')
                if result.returncode:
                    raise RuntimeError(f'BUILD_FAILED: {command[0]} returned {result.returncode}')

            latex = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'main.tex']
            run(latex)
            run(['bibtex', 'main'])
            for pass_number in range(1, 7):
                run(latex)
                latex_log = (build / 'main.log').read_text(encoding='utf-8', errors='replace')
                rerun = any('Warning' in line and any(word in line.lower()
                            for word in ('undefined', 'rerun')) for line in latex_log.splitlines())
                if not rerun:
                    break
            else:
                raise RuntimeError('Cross-references did not stabilize within six final passes')
            blocked = [line for line in latex_log.splitlines() if 'Overfull' in line or
                       ('Warning' in line and 'too large' in line.lower())]
            if blocked:
                raise RuntimeError('PDF_REQUIRES_REVIEW:\n' + '\n'.join(blocked))
            pdf_bytes = (build / 'main.pdf').read_bytes()
            if not pdf_bytes.startswith(b'%PDF-') or not pdf_bytes.rstrip().endswith(b'%%EOF'):
                raise RuntimeError('PDF is incomplete: header or final EOF marker missing')
            if shutil.which('pdfinfo') is not None:
                run(['pdfinfo', 'main.pdf'])
            # Only completed output is copied back; transient TeX auxiliaries are never reused.
            for name in ('main.pdf', 'main.log', 'main.bbl'):
                (root / name).write_bytes((build / name).read_bytes())
    except RuntimeError as exc:
        raise SystemExit(f'{exc}; see {root / "BUILD.log"}') from exc
    finally:
        with (root / 'BUILD.log').open('w', encoding='utf-8', newline='\n') as handle:
            handle.write(''.join(logs))
    print('PDF_BUILD_PASS:', root / 'main.pdf')
    print('Compilation, stable references, selected log checks and PDF completeness verified.')
    print('Visual and scientific review are recorded separately.')


if __name__ == '__main__':
    main()
