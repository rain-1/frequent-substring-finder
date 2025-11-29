import unittest

from frequent_phrases import find_repeated_phrases


class TestFrequentPhrases(unittest.TestCase):
    def test_basic_repetition(self):
        text = (
            "In this section we show that X. "
            "In this section we show that Y. "
            "In this section we show that Z."
        )
        results = find_repeated_phrases(
            text, min_words=4, max_words=7, min_count=2, normalize_case=True
        )
        self.assertEqual(len(results), 1)
        occurrence = results[0]
        self.assertEqual(occurrence.count, 3)
        self.assertEqual(occurrence.phrase, "in this section we show that")
        self.assertEqual(sorted(occurrence.sentence_indices), [0, 1, 2])

    def test_shadowing_maximality(self):
        text = "a b c d e\na b c d f\na b c d g"
        results = find_repeated_phrases(text, min_words=2, max_words=4, min_count=3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].phrase, "a b c d")
        self.assertEqual(results[0].count, 3)

    def test_independent_shorter_phrase(self):
        text = "a b c d e\na b c d f\na b c g h"
        results = find_repeated_phrases(text, min_words=2, max_words=4, min_count=2)
        phrases = {occ.phrase: occ for occ in results}
        self.assertIn("a b c", phrases)
        self.assertIn("a b c d", phrases)
        self.assertEqual(phrases["a b c"].count, 3)
        self.assertEqual(phrases["a b c d"].count, 2)

    def test_sentence_indices_mapping(self):
        text = "First overlap here. Another overlap here."
        results = find_repeated_phrases(text, min_words=2, max_words=3, min_count=2)
        self.assertEqual(len(results), 1)
        occurrence = results[0]
        self.assertEqual(occurrence.phrase, "overlap here")
        self.assertEqual(sorted(occurrence.sentence_indices), [0, 1])

    def test_punctuation_and_case_handling(self):
        text = "Hello, hello! HELLO hello."
        results = find_repeated_phrases(
            text, min_words=1, max_words=2, min_count=2, normalize_case=True
        )
        phrases = {occ.phrase: occ for occ in results}
        self.assertIn("hello", phrases)
        self.assertGreaterEqual(phrases["hello"].count, 2)


if __name__ == "__main__":
    unittest.main()
