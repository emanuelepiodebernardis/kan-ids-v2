#!/usr/bin/env python3
"""Check bilingual publication tables against retained numerical evidence; no training."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT.parent / 'experiments/hardware_validation_20260916'


def read(path):
    return path.read_text(encoding='utf-8')


def rows(text):
    return [line for line in text.splitlines()
            if '&' in line and re.match(r'\s*\$?\d', line.split('&', 1)[1])]


def cells(row):
    return row.split('&')[1:]


def numbers(cell):
    return re.findall(r'\d+(?:\.\d+)?', cell)


def rounded_equal(printed, value):
    decimals = len(printed.split('.')[1]) if '.' in printed else 0
    assert abs(float(printed) - value) <= 0.500001 * 10 ** (-decimals), (printed, value)


def main():
    hw = json.loads(read(EVIDENCE / 'hw500/HW500_PUBLICATION_DATA.json'))
    pair = json.loads(read(EVIDENCE / 'paired_software/PAIR_STAGE2_SUMMARY.json'))
    hw_rows = []
    pair_rows = []
    ram_evidence = EVIDENCE / 'ram500/evidence'
    mega = {r['model']: r for r in json.loads(read(ram_evidence / 'MEGA_RAM_SUMMARY_AS_RECORDED.json'))['rows']}
    for language in ('ieee_access', 'ru'):
        paper = ROOT / language
        htext = read(paper / 'inputs/hw500_energy.tex')
        if language == 'ru':
            htext += read(paper / 'inputs/hw500_table.tex')
        hrows = rows(htext)
        assert len(hrows) == 5, (language, len(hrows))
        for row, model in zip(hrows, ('coeff', 'lut', 'mlp', 'kanml', 'dt5')):
            c = cells(row)
            assert len(c) == 4
            for offset, board in ((0, 'mega'), (2, 'c3')):
                datum = next(v for v in hw['boards'][board]['means'] if v['model'] == model)
                rounded_equal(numbers(c[offset])[0], datum['latency_us_mean'])
                mean, sd = numbers(c[offset + 1])
                rounded_equal(mean, datum['USB_energy_uJ_mean'])
                rounded_equal(sd, datum['USB_energy_uJ_sample_SD'])
        hw_rows.append([numbers(r.split('&', 1)[1]) for r in hrows])
        prows = rows(read(paper / 'inputs/pair_stage2.tex'))
        assert len(prows) == 4
        for row, model in zip(prows, ('kan', 'dt5', 'mlp16', 'gam')):
            # Model names in saved reports are stable; unsupported names fail explicitly.
            metrics = pair['models'][model]['metrics']
            for cell, key in zip(cells(row), ('balanced_accuracy', 'f1', 'fpr', 'tpr', 'auroc')):
                mean, sd = numbers(cell)
                scale = 100 if key in ('fpr', 'tpr') else 1
                rounded_equal(mean, metrics[key]['mean'] * scale)
                rounded_equal(sd, metrics[key]['sample_sd_ddof1'] * scale)
        pair_rows.append([numbers(r.split('&', 1)[1]) for r in prows])
        ram = read(paper / 'inputs/ram500.tex')
        assert '15312' in re.sub(r'[^0-9]', '', ram), language
        assert '1584' in ram and '1504' in ram and '335' in ram
        rrows = rows(ram)
        assert len(rrows) == 5
        models = ('coeff', 'lut', 'kanml', 'mlp', 'dt5') if language == 'ieee_access' else ('coeff', 'lut', 'mlp', 'kanml', 'dt5')
        for row, model in zip(rrows, models):
            fields = cells(row)
            assert int(numbers(fields[0])[0]) == mega[model]['stack_observed_bytes_maximum_observed']
            observed = []
            for repeat in (1, 2, 3):
                record = json.loads(read(ram_evidence / f'c3_records/repeat{repeat:02d}_{model}/RUN_RECORD.json'))
                observed.append(int(record['result']['ram']['loop_stack_observed_used_bytes']))
            printed = [int(n) for cell in fields[1:] for n in numbers(cell)]
            expected = observed if language == 'ru' else sorted(set(observed))
            assert printed == expected, (language, model, printed, expected)
        all_tex = '\n'.join(read(p) for p in paper.rglob('*.tex'))
        assert '7.58' in all_tex or '7{,}58' in all_tex
        assert 'EspressifFreeRTOS447' in all_tex and 'EspressifHeap447' in all_tex
    assert hw_rows[0] == hw_rows[1], 'EN/RU HW500 tables diverged'
    assert pair_rows[0] == pair_rows[1], 'EN/RU pair tables diverged'
    print('PUBLICATION_V015_PASS: paired-model and HW500 tables agree with retained metrics in both languages; RAM scopes require scientific review.')


if __name__ == '__main__':
    main()
