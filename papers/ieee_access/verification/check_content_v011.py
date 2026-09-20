#!/usr/bin/env python3
"""Check current numerical tables, author order and optional legacy-source preservation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile


def check(root, baseline=None):
    main = (root / 'main.tex').read_text(encoding='utf-8')
    energy = (root / 'inputs/energy_pilots.tex').read_text(encoding='utf-8')
    values = json.loads((root / 'evidence/energy_pilot_results.json').read_text(encoding='utf-8'))
    model_names = {'KAN coefficients': 'coeff', 'KAN sampled LUT': 'lut',
                   'Multilayer KAN': 'kanml', 'MLP16': 'mlp', 'DT5': 'dt5'}
    rows = re.findall(r'^(KAN coefficients|KAN sampled LUT|Multilayer KAN|MLP16|DT5) & (.+?)\\\\$', energy, re.M)
    checked_cells = 0
    for name, cells in rows[:5]:
        model = model_names[name]
        means = {x['board']: x for x in values['means'] if x['model'] == model}
        m, c = means['Mega 2560'], means['ESP32-C3']
        expected = [f"{m['mean_call_us']:.3f}", f"{m['mean_power_W']*1000:.2f}",
                    f"{m['energy_uJ_per_call']:.4f}", f"{c['mean_call_us']:.4f}",
                    f"{c['mean_power_W']*1000:.2f}", f"{c['energy_uJ_per_call']:.4f}"]
        assert [x.strip() for x in cells.split('&')] == expected, name
        checked_cells += len(expected)
    assert checked_cells == 30
    memory = {
        'coeff': [22300,841,267790,15888,283856],
        'lut': [42060,841,287962,15888,300016],
        'kanml': [28266,841,273118,15904,285248],
        'mlp': [22310,841,268144,15888,283728],
        'dt5': [23232,917,269486,15968,283520],
    }
    for name, cells in rows[5:]:
        assert [int(x.strip().replace(',','')) for x in cells.split('&')] == memory[model_names[name]], name
        checked_cells += 5
    assert checked_cells == 55
    authors = json.loads((root/'AUTHOR_METADATA.json').read_text(encoding='utf-8'))
    names = ['Oleksandr Kuznetsov', 'Emanuele Pio De Bernardis', 'Emanuele Frontoni']
    assert [x['name'] for x in authors['authors']] == names
    assert 'coauthor agreement is not asserted' in authors['author_order_status']
    assert 'pdfauthor={' + '; '.join(names) + '}' in main
    assert 'Kuznetsov et al.' in main
    assert all(main.index(r'\uppercase{' + names[i] + '}') < main.index(r'\uppercase{' + names[i+1] + '}') for i in range(2))
    assert all(main.index(']{' + names[i] + '}') < main.index(']{' + names[i+1] + '}') for i in range(2))
    assert 'Current and energy were not measured' not in main
    assert 'Peak runtime memory remains unmeasured.' in main
    assert 'All five ESP32-C3 pilot applications additionally reserve' in main
    assert '0.99617' in main
    assert json.loads((root/'evidence/arch_selection_scelta.json').read_text(encoding='utf-8'))['scelte']['KAN(cat,ML)']['soglia_1se'] == 0.99617
    for item in json.loads((root/'figures/PHOTO_PROVENANCE.json').read_text(encoding='utf-8')):
        data = (root/'figures'/item['paper_asset']).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item['sha256']
        assert len(data) == item['bytes']
    original_tables = None
    if baseline:
        with zipfile.ZipFile(baseline) as old:
            tables = []
            for name in old.namelist():
                if name == 'main.tex' or (name.startswith('inputs/') and name.endswith('.tex')):
                    before = old.read(name).decode('utf-8')
                    after = (root/name).read_text(encoding='utf-8')
                    bodies = re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', before, re.S)
                    assert all(body in after for body in bodies), name
                    tables.extend(bodies)
                    if name.startswith('inputs/'):
                        assert before == after, name
            original_tables = len(tables)
            assert original_tables == 14
    return {'status':'PASS', 'version':'0.11.0', 'new_numeric_cells_verified':checked_cells,
            'legacy_tabular_bodies_preserved':original_tables,
            'proposed_author_order':names,
            'photo_bytes_match_supplied_hashes':True,
            'scope':'Manuscript consistency, not new model fitting, device execution or absolute calibration'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--baseline', type=Path)
    args = p.parse_args()
    print(json.dumps(check(args.root,args.baseline),ensure_ascii=False,indent=2))
