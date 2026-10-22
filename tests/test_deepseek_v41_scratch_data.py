import unittest

from archlab.preprocessing.v41_scratch import document_windows


class ScratchDataTests(unittest.TestCase):

    def test_document_targets_covered_once(self):
        tokens = list(range(11))
        covered = []
        for start, count in document_windows(tokens, 4):
            covered.extend(tokens[start + 1 : start + count + 1])
        self.assertEqual(covered, tokens[1:])
