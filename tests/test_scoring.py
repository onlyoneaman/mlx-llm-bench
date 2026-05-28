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


# ---------- multi-seed aggregation ----------

class TestStatsMultiSeed:
    def _rows(self, results_per_ex):
        """results_per_ex: dict {(task, i): [correct_bool, ...] (one per seed)}"""
        rows = []
        for (task, i), corrects in results_per_ex.items():
            for seed_idx, c in enumerate(corrects, 1):
                rows.append({
                    "task": task, "i": i, "correct": c, "format_ok": True,
                    "time_s": 0.1, "seed": seed_idx,
                })
        return rows

    def test_single_seed_acc_unchanged(self):
        from mlx_llm_bench.utils import stats
        rows = self._rows({("sentiment", 0): [True], ("sentiment", 1): [False]})
        s = stats(rows)
        assert s["acc"] == 50.0
        assert s["n"] == 2
        assert s["n_calls"] == 2
        assert s["n_seeds"] == 1

    def test_majority_vote_three_seeds(self):
        from mlx_llm_bench.utils import stats
        # ex 0: 2/3 correct → pass
        # ex 1: 1/3 correct → fail
        # ex 2: 3/3 correct → pass
        rows = self._rows({
            ("sentiment", 0): [True, True, False],
            ("sentiment", 1): [True, False, False],
            ("sentiment", 2): [True, True, True],
        })
        s = stats(rows)
        # 2 of 3 examples pass majority vote
        assert s["acc"] == pytest.approx(66.7, abs=0.1)
        assert s["n"] == 3  # examples
        assert s["n_calls"] == 9  # 3 examples × 3 seeds
        assert s["n_seeds"] == 3

    def test_two_seed_tie_fails(self):
        # With 2 seeds, sum > 1 means BOTH must be correct (1/2 doesn't pass).
        from mlx_llm_bench.utils import stats
        rows = self._rows({
            ("sentiment", 0): [True, False],
        })
        s = stats(rows)
        assert s["correct"] == 0

    def test_strict_acc_diverges_when_format_fails(self):
        from mlx_llm_bench.utils import stats
        rows = [
            # correct + format_ok → both pass
            {"task": "sentiment", "i": 0, "correct": True, "format_ok": True, "time_s": 0.1, "seed": 1},
            # correct via free-form fallback → acc passes, strict_acc fails
            {"task": "sentiment", "i": 1, "correct": True, "format_ok": False, "time_s": 0.1, "seed": 1},
            # wrong both ways
            {"task": "sentiment", "i": 2, "correct": False, "format_ok": False, "time_s": 0.1, "seed": 1},
        ]
        s = stats(rows)
        assert s["correct"] == 2
        assert s["strict_correct"] == 1
        assert s["acc"] > s["strict_acc"]


# ---------- dataset SHA invariance ----------

class TestDatasetShaInvariance:
    def test_sha_invariant_to_validator_key_order(self, tmp_path):
        from mlx_llm_bench.rescore import load_dataset_with_validators
        import hashlib

        data_path = tmp_path / "data.json"
        v1 = tmp_path / "v_a.json"
        v2 = tmp_path / "v_b.json"
        data_path.write_text(json.dumps([
            {"task": "ifeval", "text": "Say YES", "label": "ifeval", "difficulty": "easy"},
            {"task": "ifeval", "text": "Count 3 words.", "label": "ifeval", "difficulty": "easy"},
        ]))
        # Same content, different key ordering
        v1.write_text('{"0": [{"type":"contains_word","word":"YES"}], "1": [{"type":"word_count_exact","n":3}]}')
        v2.write_text('{"1": [{"type":"word_count_exact","n":3}], "0": [{"type":"contains_word","word":"YES"}]}')

        d1 = load_dataset_with_validators(data_path, v1)
        d2 = load_dataset_with_validators(data_path, v2)
        s1 = hashlib.sha256(json.dumps(d1, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        s2 = hashlib.sha256(json.dumps(d2, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        assert s1 == s2

    def test_ifeval_with_no_validators_raises(self, tmp_path):
        from mlx_llm_bench.rescore import load_dataset_with_validators
        data_path = tmp_path / "data.json"
        # IFEval row but no validators file
        data_path.write_text(json.dumps([
            {"task": "ifeval", "text": "Say YES", "label": "ifeval", "difficulty": "easy"},
        ]))
        with pytest.raises(ValueError, match="no validators"):
            load_dataset_with_validators(data_path)

    def test_validator_pointing_at_non_ifeval_raises(self, tmp_path):
        from mlx_llm_bench.rescore import load_dataset_with_validators
        data_path = tmp_path / "data.json"
        v_path = tmp_path / "v.json"
        data_path.write_text(json.dumps([
            {"task": "sentiment", "text": "x", "label": "positive", "difficulty": "easy"},
        ]))
        v_path.write_text('{"0": [{"type":"contains_word","word":"x"}]}')
        with pytest.raises(ValueError, match="not ifeval"):
            load_dataset_with_validators(data_path, v_path)


# ---------- JSON validator robustness ----------

class TestJsonValidatorRobustness:
    def test_nested_array_does_not_match_inner_brackets_only(self):
        from mlx_llm_bench.rescore import _v_json_array_length
        # Old non-greedy regex `\[.*?\]` would match `[[1,2]` and fail to parse.
        assert _v_json_array_length("[[1,2],[3,4]]", {"n": 2}) is True

    def test_multiple_json_objects_in_response(self):
        from mlx_llm_bench.rescore import _v_json_has_keys
        # Old greedy regex `\{.*\}` would span across both objects.
        assert _v_json_has_keys('{"a":1} ignore {"b":2}', {"keys": ["a"]}) is True

    def test_json_in_freeform_response(self):
        from mlx_llm_bench.rescore import _v_json_has_keys
        raw = 'Here is the result:\n{"name": "Ada", "age": 36}\nHope that helps!'
        assert _v_json_has_keys(raw, {"keys": ["name", "age"]}) is True

    def test_brackets_inside_strings_dont_confuse_extractor(self):
        from mlx_llm_bench.rescore import _v_json_array_length
        assert _v_json_array_length('["a", "[b]", "c"]', {"n": 3}) is True


# ---------- HTTP error tagging ----------

class TestHttpErrorTagging:
    def test_http_error_is_scored_wrong_and_not_format_ok(self):
        # Mirror the _score branch in runner.py
        from mlx_llm_bench.runner import _score
        ex = {"task": "sentiment", "label": "positive"}
        pred, fmt_ok, correct, vres = _score(ex, "__HTTP_ERROR__: connection refused", fmt="json")
        assert correct is False
        assert fmt_ok is False
        # Even if the error text happens to contain "positive" by accident,
        # we don't want to count it as a correct answer.

    def test_generic_error_tag_also_handled(self):
        from mlx_llm_bench.runner import _score
        ex = {"task": "sentiment", "label": "positive"}
        pred, fmt_ok, correct, vres = _score(ex, "__ERROR__: timeout", fmt="json")
        assert correct is False
        assert fmt_ok is False
