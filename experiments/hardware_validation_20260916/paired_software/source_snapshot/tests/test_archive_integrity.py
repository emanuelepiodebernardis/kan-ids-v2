"""Regression for the truncated-score artifact found during the full run."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from run_stage2 import archive_run


class ArchiveIntegrityTests(unittest.TestCase):
    def prepare(self, directory, status):
        run = Path(directory) / 'run'
        run.mkdir()
        (run / 'RUN_RECORD.json').write_text(json.dumps({'status': status}), encoding='utf-8')
        # Keep the gzip header/data but remove its CRC/length trailer.
        (run / 'test_scores.csv.gz').write_bytes(gzip.compress(b'row_id\n1\n')[:-8])
        return run

    def test_complete_run_cannot_accept_a_truncated_score_file(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.prepare(directory, 'STUDY_COMPLETE_FIVE_SEEDS')
            with self.assertRaises(EOFError):
                archive_run(run)

    def test_failed_run_preserves_partial_evidence_for_diagnosis(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.prepare(directory, 'BLOCKED_OR_FAILED')
            self.assertTrue(archive_run(run).is_file())


if __name__ == '__main__':
    unittest.main()
