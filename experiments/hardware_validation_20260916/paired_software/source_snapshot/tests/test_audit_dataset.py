import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_dataset import NUMERIC, CATEGORICAL, audit, pair_split


class AuditTests(unittest.TestCase):
    def test_reverse_direction_and_label_independence(self):
        self.assertEqual(pair_split('192.0.2.1', '198.51.100.2'),
                         pair_split('198.51.100.2', '192.0.2.1'))
        self.assertNotEqual(pair_split('a,b', 'c')[1], pair_split('a', 'b,c')[1])

    def test_duplicate_groups_preserved_and_source_not_changed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'input.csv'
            header = ['src_ip', 'dst_ip'] + NUMERIC + CATEGORICAL + ['label', 'type']
            rows = []
            found = {}
            for i in range(500):
                a, b = '192.0.2.' + str(i), '198.51.100.1'
                s, h = pair_split(a, b)
                found.setdefault(s, (a, b))
                if len(found) == 3:
                    break
            for split, (a, b) in found.items():
                for label, name in [(0, 'normal'), (1, 'dos')]:
                    row = dict.fromkeys(header, '0')
                    row.update(src_ip=a, dst_ip=b, label=str(label), type=name, proto='tcp')
                    row['duration'] = str(label + 1)
                    rows.extend([row, dict(row)])
            with path.open('w', encoding='utf-8', newline='') as f:
                w = csv.DictWriter(f, fieldnames=header, lineterminator='\n')
                w.writeheader()
                w.writerows(rows)
            before = path.read_bytes()
            report = audit(path, root / 'new', require_canonical=False)
            self.assertTrue(report['pilot_allowed'])
            self.assertEqual(report['full_row_duplicate_excess'], 6)
            self.assertTrue(all(x['shared_full_row_keys'] == 0 for x in report['overlap'].values()))
            self.assertFalse(report['claims']['host_disjoint'])
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                audit(path, root / 'new', require_canonical=False)

    def test_wrong_dataset_refused_before_output_creation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'wrong.csv'
            path.write_text('label\n0\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'SOURCE_SHA256_MISMATCH'):
                audit(path, root / 'out')
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
