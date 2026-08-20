from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from incident_assistant.evaluation.loghub import benchmark_hdfs_log


class LogHubBenchmarkTests(TestCase):
    def test_benchmark_parses_hdfs_lines_and_reports_unmatched_input(self) -> None:
        lines = (
            "081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block "
            "blk_-1 from /10.0.0.1\n"
            "081109 203519 143 WARN dfs.FSNamesystem: Slow block blk_2\n"
            "not an HDFS log line\n"
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "HDFS_2k.log"
            path.write_text(lines, encoding="utf-8")
            result = benchmark_hdfs_log(path)

        self.assertEqual(result["total_lines"], 3)
        self.assertEqual(result["parsed_lines"], 2)
        self.assertAlmostEqual(result["parse_rate"], 2 / 3)
        self.assertEqual(result["severity_counts"], {"INFO": 1, "WARN": 1})
        self.assertEqual(result["unique_components"], 2)
        self.assertEqual(result["unique_block_ids"], 2)

    def test_benchmark_rejects_non_positive_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            benchmark_hdfs_log(Path("unused"), max_lines=0)
