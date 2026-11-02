"""Copy validated numerical assets into neutral submission filenames."""
from pathlib import Path
import shutil
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'review_extension/manuscript'
def neutral(text):
    return text.replace('review2','diagnostic_a').replace('review3','diagnostic_b').replace('CAL-role256','ROLE256').replace('CAL-role64','ROLE64').replace('CAL256','ALL256').replace('CAL64','ROLE64')
for folder in ['figures','tables']:
    for p in (SOURCE/folder).iterdir():
        if p.suffix not in {'.tex','.pdf','.png'}:continue
        q=HERE/folder/neutral(p.name);q.parent.mkdir(exist_ok=True)
        if p.suffix=='.tex':q.write_text(neutral(p.read_text()))
        else:shutil.copyfile(p,q)
for name in ['review2_audit_summary.tex','review2_findings.tex','review2_sensitivity_summary.tex','review2_matched_summary.tex']:
    (HERE/neutral(name)).write_text(neutral((SOURCE/name).read_text()))
print('Numerical assets refreshed; manuscript prose unchanged.')
