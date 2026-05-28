"""Tests for the canonical scoring + stats functions.

Run: pytest tests/

These exist to catch regressions in parse_answer / validate_ifeval /
wilson_ci / mcnemar_p after refactors. Reference values computed by hand
where possible.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlx_llm_bench.rescore import (  # noqa: E402
    clean_raw,
    load_dataset_with_validators,
    parse_answer,
    validate_ifeval,
    word_count,
)
from mlx_llm_bench.utils import mcnemar_p, wilson_ci  # noqa: E402


# ---------- parse_answer ----------

class TestParseAnswerText:
    def test_simple_label_word(self):
        pred, fmt_ok = parse_answer("positive", "sentiment", fmt="text")
        assert pred == "positive"
        assert fmt_ok is True

    def test_label_in_sentence(self):
        pred, fmt_ok = parse_answer("This is positive sentiment", "sentiment", fmt="text")
        assert pred == "positive"
        assert fmt_ok is True

    def test_no_valid_label_returns_first_word(self):
        pred, fmt_ok = parse_answer("ham nor spam", "sentiment", fmt="text")
        # 'ham' / 'spam' aren't sentiment labels; fall back to first word
        assert pred == "ham"
        assert fmt_ok is False

    def test_strips_think_block(self):
        pred, fmt_ok = parse_answer("<think>blah blah</think>negative", "sentiment", fmt="text")
        assert pred == "negative"

    def test_strips_special_tokens(self):
        # The fix for Qwen2.5-Coder leaking <|im_end|>
        pred, fmt_ok = parse_answer("positive<|im_end|>\n", "sentiment", fmt="text")
        assert pred == "positive"


class TestParseAnswerJson:
    def test_valid_json_label(self):
        pred, fmt_ok = parse_answer('{"label": "positive"}', "sentiment", fmt="json")
        assert pred == "positive"
        assert fmt_ok is True

    def test_json_with_extra_whitespace(self):
        pred, fmt_ok = parse_answer('  {"label":  "world"}  ', "topic", fmt="json")
        assert pred == "world"
        assert fmt_ok is True

    def test_freeform_text_fallback_marks_not_ok(self):
        # Model said the right thing but ignored JSON format
        pred, fmt_ok = parse_answer("the answer is positive", "sentiment", fmt="json")
        assert pred == "positive"
        assert fmt_ok is False

    def test_json_with_invalid_label_marks_not_ok(self):
        pred, fmt_ok = parse_answer('{"label": "neutral"}', "sentiment", fmt="json")
        assert pred == "neutral"
        assert fmt_ok is False


# ---------- validators ----------

class TestValidators:
    def _check(self, raw, validators, expected_pass):
        all_pass, _ = validate_ifeval(raw, validators)
        assert all_pass is expected_pass

    def test_word_count_exact_pass(self):
        self._check("one two three", [{"type": "word_count_exact", "n": 3}], True)

    def test_word_count_exact_fail(self):
        self._check("one two", [{"type": "word_count_exact", "n": 3}], False)

    def test_word_count_em_dash_counts_as_two_words(self):
        # The em-dash fix — `\b[\w']+\b` regex correctly splits compound words
        # joined by em-dash. str.split() would have returned 11 here, not 12.
        # This is the actual Qwen3-8B response on data.json's "describe a
        # close friend in exactly 12 words" item.
        text = "Lively, loyal, and always there—my confidant through life's ups and downs."
        all_pass, _ = validate_ifeval(text, [{"type": "word_count_exact", "n": 12}])
        assert all_pass is True

    def test_word_count_min(self):
        self._check("one two three four", [{"type": "word_count_min", "n": 3}], True)
        self._check("one two", [{"type": "word_count_min", "n": 3}], False)

    def test_word_count_max(self):
        self._check("one two", [{"type": "word_count_max", "n": 3}], True)
        self._check("one two three four", [{"type": "word_count_max", "n": 3}], False)

    def test_contains_word_case_insensitive(self):
        self._check("Binary is the answer", [{"type": "contains_word", "word": "binary"}], True)
        self._check("decimal works too", [{"type": "contains_word", "word": "binary"}], False)

    def test_not_contains_word(self):
        self._check("decimal only", [{"type": "not_contains_word", "word": "binary"}], True)
        self._check("uses binary code", [{"type": "not_contains_word", "word": "binary"}], False)

    def test_not_contains_letter(self):
        self._check("sturdy ash bough", [{"type": "not_contains_letter", "letter": "e"}], True)
        self._check("hello world", [{"type": "not_contains_letter", "letter": "e"}], False)

    def test_regex_match_all_caps(self):
        pattern = r"^[A-Z0-9\s\.,!?'\"\-:;()]+$"
        self._check("PARIS", [{"type": "regex_match", "pattern": pattern}], True)
        self._check("Paris", [{"type": "regex_match", "pattern": pattern}], False)

    def test_starts_with(self):
        self._check("Indeed it is so.", [{"type": "starts_with", "prefix": "Indeed"}], True)
        self._check("Verily it is.", [{"type": "starts_with", "prefix": "Indeed"}], False)

    def test_ends_with(self):
        self._check("That was fast!", [{"type": "ends_with", "suffix": "!"}], True)
        self._check("That was fast.", [{"type": "ends_with", "suffix": "!"}], False)

    def test_json_array_length(self):
        self._check('["red","green","blue"]', [{"type": "json_array_length", "n": 3}], True)
        self._check('["red","green"]', [{"type": "json_array_length", "n": 3}], False)

    def test_json_has_keys(self):
        self._check('{"name":"Ada","age":36}', [{"type": "json_has_keys", "keys": ["name", "age"]}], True)
        self._check('{"name":"Ada"}', [{"type": "json_has_keys", "keys": ["name", "age"]}], False)

    def test_exact_word_set(self):
        # exact_word_set strips non-letters then case-insensitive matches
        self._check("YES", [{"type": "exact_word_set", "options": ["YES", "NO"]}], True)
        self._check("yes.", [{"type": "exact_word_set", "options": ["YES", "NO"]}], True)
        self._check("maybe", [{"type": "exact_word_set", "options": ["YES", "NO"]}], False)

    def test_multiple_validators_all_must_pass(self):
        validators = [
            {"type": "contains_word", "word": "ocean"},
            {"type": "word_count_min", "n": 5},
        ]
        self._check("the ocean is vast and deep", validators, True)
        self._check("ocean", validators, False)  # word count fails
        self._check("the sea is vast and deep", validators, False)  # word missing


# ---------- Wilson CI ----------

class TestWilsonCI:
    def test_zero_n(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_50_of_100_centered_on_50(self):
        lo, hi = wilson_ci(50, 100)
        # Reference value for Wilson 95% CI on 50/100 ≈ (40.4, 59.6)
        assert 40.0 <= lo <= 41.0
        assert 59.0 <= hi <= 60.0

    def test_all_correct_upper_bounded_by_100(self):
        lo, hi = wilson_ci(100, 100)
        assert hi <= 100.0
        assert lo > 0.0  # not perfectly tight at 100

    def test_none_correct_lower_bounded_by_0(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0
        assert hi > 0.0  # bounded by the failure rate uncertainty


# ---------- McNemar ----------

class TestMcNemar:
    def test_no_discordance(self):
        # b == c == 0: no evidence of difference
        assert mcnemar_p(0, 0) == 1.0

    def test_equal_discordance(self):
        # b == c: maximum p-value
        p = mcnemar_p(5, 5)
        assert p > 0.9

    def test_9_vs_1_significant(self):
        # 9 wins / 1 loss across 10 paired discordant trials → ~0.021, significant
        p = mcnemar_p(9, 1)
        assert 0.01 < p < 0.03

    def test_8_vs_2_not_significant(self):
        # 8/2 split → ~0.109, not significant at α=0.05
        p = mcnemar_p(8, 2)
        assert 0.10 < p < 0.12

    def test_capped_at_1(self):
        # Anything where 2 * one_side > 1 should be clamped
        p = mcnemar_p(3, 3)
        assert p == 1.0


# ---------- load_dataset_with_validators ----------

class TestLoadDataset:
    def test_merges_validators_for_ifeval_rows(self, tmp_path):
        data_path = tmp_path / "data.json"
        validators_path = tmp_path / "ifeval_validators.json"
        data_path.write_text(json.dumps([
            {"task": "sentiment", "text": "good", "label": "positive", "difficulty": "easy"},
            {"task": "ifeval", "text": "Say YES", "label": "ifeval", "difficulty": "easy"},
        ]))
        validators_path.write_text(json.dumps({
            "1": [{"type": "exact_word_set", "options": ["YES"]}],
        }))
        data = load_dataset_with_validators(data_path)
        assert data[0].get("validators") is None
        assert data[1]["validators"] == [{"type": "exact_word_set", "options": ["YES"]}]

    def test_works_without_validators_file(self, tmp_path):
        data_path = tmp_path / "data.json"
        data_path.write_text(json.dumps([
            {"task": "sentiment", "text": "good", "label": "positive", "difficulty": "easy"},
        ]))
        data = load_dataset_with_validators(data_path)
        assert len(data) == 1
        assert data[0].get("validators") is None


# ---------- clean_raw / word_count ----------

class TestUtilities:
    def test_clean_raw_strips_think(self):
        assert clean_raw("<think>hidden</think>visible") == "visible"

    def test_clean_raw_strips_special_tokens(self):
        assert clean_raw("answer<|im_end|>\n") == "answer\n"

    def test_word_count_treats_em_dash_as_separator(self):
        # The bug fix: str.split() gives 5 here, word_count gives 6.
        assert word_count("there—my confidant in life's ups downs") == 7

    def test_word_count_handles_apostrophes_as_part_of_word(self):
        # "life's" should count as 1, not 2
        assert word_count("life's good") == 2
