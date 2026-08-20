import unittest

from incident_assistant.infrastructure.extraction import RegexMachineIdExtractor


class RegexMachineIdExtractorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = RegexMachineIdExtractor()

    def test_extracts_and_normalizes_machine_id(self) -> None:
        result = self.extractor.extract("Coffee from cm-1001 is watery")
        self.assertEqual(result, "CM-1001")

    def test_returns_none_when_id_is_missing(self) -> None:
        self.assertIsNone(self.extractor.extract("The coffee tastes watery"))


if __name__ == "__main__":
    unittest.main()
