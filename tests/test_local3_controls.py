import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import jsonschema

from probity_benchmark.extract_iteration import intervals
from probity_benchmark.evaluate_debate_transfer import argument_mapping
from probity_benchmark.local3_bundle import segment_scores
from probity_benchmark.local3_annotate_units import display_units, reconcile_unit
from probity_benchmark.local3_diagnose import audit
from probity_benchmark.local3_fit import fit_head, paired_metrics
from probity_benchmark.local3_fit import main as fit_main
from probity_benchmark.local3_natural_twins import check_balanced_pairs, replace_source
from probity_benchmark.local3_mixed_twins import mutate
from probity_benchmark.local3_project_cache import project
from probity_benchmark.local3_score_units import score_cache
from probity_benchmark.local3_evaluate_units import paired_delta
from probity_benchmark.local3_evaluate_units import main as evaluate_units_main
from probity_benchmark.preflight import sha256
from probity_benchmark.local3_secondary_clauses import opening_ids, complete_payload
from probity_benchmark.local3_retain_cache import (
    retain,
    verify_retained,
    tensor_receipts,
)
from probity_benchmark.local3_verify_twins import validate_checks
from probity_benchmark.qwen_extract import LAYERS


class NaturalTwinControlTests(unittest.TestCase):
    def test_fresh_join_preserves_units_and_measures_frozen_threshold(self):
        raw = json.dumps({"argument": "Ari stays. Bea leaves.", "answer": "A"})
        _, units = display_units(raw, ["sentence"])
        predicted = {
            "id": "row",
            "group": "parent",
            "participant_model": "tiny",
            "response_sha256": sha256(raw),
            "units": [
                {**u, "scores": {"local3": float(i), "response_broadcast": 0.6}}
                for i, u in enumerate(units)
            ],
        }
        annotated = {
            "id": "row",
            "group": "parent",
            "participant_model": "tiny",
            "response": raw,
            "annotations": [
                {"unit": u, "label": i, "rejection_reasons": []}
                for i, u in enumerate(units)
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scores, annotations = root / "scores", root / "annotations"
            scores.mkdir()
            annotations.mkdir()
            (scores / "scores.jsonl").write_text(json.dumps(predicted) + "\n")
            (scores / "manifest.json").write_text(
                json.dumps({"models": {"tiny": {"thresholds": {"local3": 0.5}}}})
            )
            (annotations / "corpus.jsonl").write_text(json.dumps(annotated) + "\n")
            (annotations / "manifest.json").write_text("{}")
            (annotations / "summary.json").write_text(
                json.dumps(
                    {
                        "errors": [],
                        "expected_rows": 1,
                        "completed_rows": 1,
                        "expected_units": 2,
                        "completed_units": 2,
                    }
                )
            )
            families = root / "families.json"
            families.write_text(json.dumps({"by_parent": {"parent": "family"}}))
            out = root / "results"
            arguments = [
                "local3_evaluate_units",
                "--scores",
                str(scores),
                "--annotations",
                str(annotations),
                "--family-map",
                str(families),
                "--out",
                str(out),
            ]
            with patch("sys.argv", arguments), redirect_stdout(io.StringIO()):
                evaluate_units_main()
            result = json.loads((out / "results.json").read_text())["cells"]["tiny"][
                "sentence"
            ]
            self.assertEqual(result["metrics"]["local3"]["flag_precision"], 1)
            self.assertEqual(
                result["metrics"]["local3"]["true_claim_false_positive_rate"], 0
            )
            self.assertEqual(
                result["metrics"]["response_broadcast"]["within_response_auroc"][
                    "parent_macro_mean"
                ],
                0.5,
            )
            annotated["response"] = raw + " "
            (annotations / "corpus.jsonl").write_text(json.dumps(annotated) + "\n")
            arguments[-1] = str(root / "invalid")
            with patch("sys.argv", arguments), self.assertRaises(ValueError):
                evaluate_units_main()

    def test_all_unit_scores_match_judge_export_and_broadcast_is_constant(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        raw = json.dumps(
            {"argument": 'Ari said "yes"; Bea left. Cy stayed.', "answer": "A"}
        )
        row = {
            "id": "r",
            "group": "p",
            "participant_model": "tiny",
            "source_episode": "e",
            "turn_index": 0,
            "source": "api_debate_silver",
            "response": raw,
        }
        segments, _ = intervals(row)
        features = torch.ones(1, len(segments), 2)
        for i, segment in enumerate(segments):
            features[0, i] = torch.tensor(
                [float(segment["start"]), float(segment["end"])]
            )
        cache = {
            "row": row,
            "segments": segments,
            "segment_mean": features,
            "segment_last": features + 1,
            "segment_lengths": [int(s["end"] - s["start"]) for s in segments],
            "response_mean": torch.tensor([[2.0, 3.0]]),
            "fibs_contrast": 0.4,
        }
        head = {
            "feature": "span_mean",
            "layer": 35,
            "weight": torch.tensor([1.0, -1.0]),
            "bias": 0.25,
        }
        saved = {
            "__contract__": {"layers": [35]},
            "local3": head,
            "random_direction": head,
        }
        old = {"__contract__": {"selected_track": "old"}, "old": head}
        response = {
            "__contract__": {"selected_feature": "response_mean"},
            "response_mean": {**head, "feature": "response_mean"},
        }
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(["Ari yes", "Bea left"])
        classifier = LogisticRegression().fit(matrix, [0, 1])
        output = score_cache(
            cache,
            saved,
            old,
            response,
            {"vectorizer": vectorizer, "classifier": classifier},
        )
        for granularity in ["sentence", "clause"]:
            _, intervals_out, expected = segment_scores(cache, head, [35], granularity)
            actual = [u for u in output["units"] if granularity in u["granularities"]]
            self.assertEqual(
                [[u["argument_start"], u["argument_end"]] for u in actual],
                intervals_out,
            )
            self.assertEqual([u["scores"]["local3"] for u in actual], expected)
        self.assertEqual(
            {u["scores"]["response_broadcast"] for u in output["units"]}, {-0.75}
        )

    def test_paired_delta_weights_story_families_before_bootstrap(self):
        y = np.asarray([0, 1, 0, 1, 0, 1])
        ids = np.asarray(["a", "a", "b", "b", "c", "c"])
        families = np.asarray(["first", "first", "first", "first", "second", "second"])
        main = np.asarray([0, 1, 0, 1, 1, 0])
        control = np.zeros(6)
        result = paired_delta(y, main, control, ids, families)
        self.assertEqual(result["families"], 2)
        self.assertEqual(result["mean"], 0.0)

    def test_fixed_units_deduplicate_bounds_without_losing_granularity(self):
        raw = json.dumps(
            {"argument": 'Ari said "yes"; Bea left. Cy stayed.', "answer": "A"}
        )
        argument, units = display_units(raw, ["sentence", "clause"])
        self.assertEqual(
            len({(u["argument_start"], u["argument_end"]) for u in units}), len(units)
        )
        shared = [u for u in units if len(u["granularities"]) == 2]
        self.assertEqual(len(shared), 1)
        for unit in units:
            self.assertEqual(raw[unit["start"] : unit["end"]], unit["text"])
            self.assertEqual(
                argument[unit["argument_start"] : unit["argument_end"]],
                unit["display_text"],
            )

    def test_unknown_or_disputed_units_never_become_truthful_negatives(self):
        unit = {"display_text": "Ari stayed."}
        supported = {
            "exact_text": unit["display_text"],
            "verdict": "supported",
            "local_label_valid": True,
            "claim_kind": "checkable_fact",
            "story_paragraph_ids": [0],
        }
        label, reasons = reconcile_unit(unit, [supported, dict(supported)])
        self.assertEqual((label, reasons), (0, []))
        unknown = {**supported, "verdict": "unknown"}
        label, reasons = reconcile_unit(unit, [supported, unknown])
        self.assertIsNone(label)
        self.assertIn("unknown", reasons)
        invalid = {**supported, "local_label_valid": False}
        self.assertIsNone(reconcile_unit(unit, [supported, invalid])[0])

    def test_mixed_augmentation_changes_one_claim_and_preserves_json_scope(self):
        argument = 'Ari said "yes". Beatrice left.'
        response = json.dumps({"argument": argument, "answer": "A"})
        _, mapping = argument_mapping(response)
        spans = []
        for i, text in enumerate(['Ari said "yes".', "Beatrice left."]):
            a = argument.index(text)
            b = a + len(text)
            start, end = mapping[a][0], mapping[b - 1][1]
            spans.append(
                {
                    "original_span_index": i,
                    "argument_start": a,
                    "argument_end": b,
                    "start": start,
                    "end": end,
                    "display_text": text,
                    "text": response[start:end],
                    "label": 0,
                }
            )
        row = {"pair_id": "test", "response": response}
        changed, remapped, audit = mutate(
            row, spans, {"Ari": "Beatrice", "Beatrice": "Ari"}
        )
        decoded = json.loads(changed)
        self.assertEqual(decoded["answer"], "A")
        self.assertEqual(sum(s["label"] for s in remapped), 1)
        self.assertEqual(
            decoded["argument"], argument.replace(audit["before"], audit["after"])
        )
        for span in remapped:
            self.assertEqual(changed[span["start"] : span["end"]], span["text"])
            self.assertEqual(
                decoded["argument"][span["argument_start"] : span["argument_end"]],
                span["display_text"],
            )

    def test_complete_cached_fit_keeps_family_split_and_text_pair_control(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            natural, twins = root / "natural", root / "twins"
            natural.mkdir()
            twins.mkdir()
            manifest = {"model": "tiny", "model_revision": "test", "layers": LAYERS}
            prediction_rows = []
            for index, split in enumerate(["train", "dev", "test"]):
                row = {
                    "id": f"original-{index}",
                    "group": f"p{index}",
                    "split": split,
                    "participant_model": "qwen/example",
                    "response": "Alpha. Beta.",
                    "spans": [
                        {"start": 0, "end": 6, "text": "Alpha.", "label": 0},
                        {"start": 7, "end": 12, "text": "Beta.", "label": 1},
                    ],
                }
                features = torch.tensor(
                    [[[-1.0, 2.0, 1.0], [1.0, -2.0, 1.0]]] * len(LAYERS)
                )
                cache = {
                    "row": row,
                    "span_mean": features,
                    "span_last": features,
                    "span_lengths": [2, 2],
                }
                torch.save(cache, natural / (row["id"] + ".pt"))
                for i, span in enumerate(row["spans"]):
                    prediction_rows.append(
                        {
                            "id": row["id"],
                            "span_index": i,
                            "text": span["text"],
                            "scores": {"text_only_tfidf": float(i)},
                        }
                    )
                if split == "test":
                    continue
                for world in [0, 1]:
                    twin = copy.deepcopy(row)
                    twin.update(
                        id=row["id"] + f"-world{world}",
                        pair_id=row["id"],
                        context_world=world,
                        source_row_id=row["id"],
                        augmentation={"kind": "single_claim_entity_swap"},
                    )
                    for i, span in enumerate(twin["spans"]):
                        span.update(label=i ^ world, original_span_index=i)
                    varied = features.clone()
                    varied[:, :, 0] *= 1 if world == 0 else -1
                    torch.save(
                        {
                            **cache,
                            "row": twin,
                            "span_mean": varied,
                            "span_last": varied,
                        },
                        twins / (twin["id"] + ".pt"),
                    )
            for directory in [natural, twins]:
                count = len(list(directory.glob("*.pt")))
                (directory / "summary.json").write_text(
                    json.dumps({"rows_expected": count, "rows_completed": count})
                )
                (directory / "manifest.json").write_text(json.dumps(manifest))
            head_path = root / "old.pt"
            torch.save(
                {
                    "combined": {
                        "feature": "span_mean",
                        "layer": 35,
                        "weight": torch.tensor([1.0, 0.0, 0.0]),
                        "bias": 0,
                    },
                    "__contract__": {**manifest, "selected_track": "combined"},
                },
                head_path,
            )
            predictions = root / "predictions.jsonl"
            predictions.write_text(
                "".join(json.dumps(row) + "\n" for row in prediction_rows)
            )
            families = root / "families.json"
            families.write_text(
                json.dumps(
                    {
                        "families": {
                            split: [{"problem_id": f"p{i}"}]
                            for i, split in enumerate(["train", "dev", "test"])
                        }
                    }
                )
            )
            out = root / "fitted"
            arguments = [
                "local3_fit",
                "--natural",
                str(natural),
                "--twins",
                str(twins),
                "--old-heads",
                str(head_path),
                "--old-predictions",
                str(predictions),
                "--family-audit",
                str(families),
                "--participant",
                "qwen/example",
                "--out",
                str(out),
            ]
            with patch("sys.argv", arguments), redirect_stdout(io.StringIO()):
                fit_main()
            results = json.loads((out / "results.json").read_text())
            self.assertEqual(len(results["candidates"]), 32)
            self.assertEqual(
                results["controls"]["frozen_iteration2_text_only"]["development"][
                    "source_twins"
                ]["parent_macro_order_accuracy"],
                0.5,
            )
            self.assertEqual(
                results["selected_development"]["source_twins"]["parent_groups"], 1
            )
            saved = torch.load(out / "heads.pt", weights_only=True)
            self.assertFalse(saved["__contract__"]["fresh_holdout_used"])

    def test_pooled_projection_rejects_source_changes_and_keeps_exact_features(self):
        row = {
            "id": "row",
            "pair_id": "pair",
            "group": "story",
            "split": "train",
            "participant_model": "test",
            "context_world": 0,
            "messages": [{"role": "user", "content": "Complete source"}],
            "response": "A. B.",
            "spans": [
                {
                    "original_span_index": i,
                    "start": start,
                    "end": end,
                    "text": text,
                    "display_text": text,
                    "label": i,
                }
                for i, (start, end, text) in enumerate([(0, 2, "A."), (3, 5, "B.")])
            ],
        }
        features = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        cache = {
            "row": row,
            "span_mean": features,
            "span_last": features + 10,
            "span_lengths": [1, 1],
            "segment_mean": features + 20,
        }
        verified = copy.deepcopy(row)
        verified["spans"] = [verified["spans"][1]]
        projected, indices = project(cache, verified)
        self.assertEqual(indices, [1])
        self.assertTrue(torch.equal(projected["span_mean"], features[:, [1]]))
        self.assertTrue(torch.equal(projected["segment_mean"], cache["segment_mean"]))
        verified["messages"][0]["content"] = "Different source"
        with self.assertRaises(ValueError):
            project(cache, verified)

    def test_adjudicator_cannot_accept_factual_verdict_as_interpretation(self):
        check = {
            "index": 0,
            "claim_kind": "interpretation_or_advocacy",
            "local_label_valid": True,
            "verdict": "supported",
            "story_paragraph_ids": [0],
        }
        with self.assertRaises(jsonschema.ValidationError):
            validate_checks({"checks": [check]}, 1, 1)
        check["claim_kind"] = "checkable_fact"
        validate_checks({"checks": [check]}, 1, 1)
        check["verdict"] = "unknown"
        check["story_paragraph_ids"] = []
        validate_checks({"checks": [check]}, 1, 1)

    def test_all_text_bundle_keeps_escaped_json_and_rejects_missing_units(self):
        argument = 'Alice said "yes"; Bob said no. Both stayed.'
        row = {
            "source": "api_debate_silver",
            "response": json.dumps({"argument": argument, "answer": "A"}),
        }
        segments, _ = intervals(row)
        features = torch.ones(1, len(segments), 2)
        cache = {"row": row, "segments": segments, "segment_mean": features}
        head = {
            "feature": "span_mean",
            "layer": 35,
            "weight": torch.tensor([1.0, 2.0]),
            "bias": -1.0,
        }
        text, spans, scores = segment_scores(cache, head, [35], "clause")
        self.assertEqual(text, argument)
        self.assertEqual("".join(text[a:b] for a, b in spans), argument)
        self.assertEqual(scores, [2.0] * len(spans))
        cache["segments"] = [s for s in segments if s["kind"] == "sentence"]
        with self.assertRaises(ValueError):
            segment_scores(cache, head, [35], "clause")

    def test_source_pair_objective_cannot_fit_identical_text_feature(self):
        rows, features = [], []
        for pair in range(4):
            for label in [0, 1]:
                rows.append(
                    {
                        "pair_id": str(pair),
                        "original_span_index": 0,
                        "group": "story" + str(pair // 2),
                        "text": "Complete identical claim " + str(pair),
                        "position": pair / 4,
                        "y": label,
                    }
                )
                features.append([100 * pair, 2 * label - 1])
        x = np.asarray(features, dtype=float)
        y = np.asarray([r["y"] for r in rows])
        absent = np.zeros(len(rows), dtype=bool)
        present = ~absent
        weight, bias = fit_head(x, y, rows, absent, present, "twins_pairwise", 0.01)
        self.assertAlmostEqual(weight[0], 0)
        self.assertGreater(weight[1], 0)
        metric = paired_metrics(rows, x @ weight + bias, present)
        self.assertEqual(metric["parent_macro_order_accuracy"], 1)
        identical = paired_metrics(rows, x[:, 0], present)
        self.assertEqual(identical["parent_macro_order_accuracy"], 0.5)

    def test_source_intervention_preserves_other_messages_and_history(self):
        messages = [
            {"role": "system", "content": "Argue for River even when it is wrong."},
            {
                "role": "user",
                "content": "Source:\nRiver said yes.\nHistory:\nRiver said yes, says the opponent.",
            },
        ]
        replaced = replace_source(
            messages, "Source:\nRiver said yes.", "Source:\nGrove said yes."
        )
        self.assertEqual(messages[0], replaced[0])
        self.assertEqual(
            replaced[1]["content"],
            "Source:\nGrove said yes.\nHistory:\nRiver said yes, says the opponent.",
        )
        self.assertEqual(
            messages[1]["content"],
            "Source:\nRiver said yes.\nHistory:\nRiver said yes, says the opponent.",
        )
        with self.assertRaises(ValueError):
            replace_source(messages, "River said yes", "Grove said yes")

    def test_context_twins_reject_label_or_text_mismatch(self):
        row = {
            "pair_id": "p",
            "group": "g",
            "context_world": 0,
            "response": "Alice spoke.",
            "spans": [{"start": 0, "end": 12, "text": "Alice spoke.", "label": 0}],
        }
        other = copy.deepcopy(row)
        other["context_world"] = 1
        other["spans"][0]["label"] = 1
        self.assertEqual(check_balanced_pairs([row, other])["claim_pairs"], 1)
        other["spans"][0]["label"] = 0
        with self.assertRaises(ValueError):
            check_balanced_pairs([row, other])
        other["spans"][0]["label"] = 1
        other["spans"][0]["text"] = "Alice lied."
        with self.assertRaises(ValueError):
            check_balanced_pairs([row, other])

    def test_position_matching_does_not_cross_responses(self):
        rows = []
        predictions = []
        for unit, label in [("a", 0), ("b", 1)]:
            rows.append(
                {
                    "id": unit,
                    "group": "same_story",
                    "split": "dev",
                    "assigned_advocacy_correct": True,
                    "response": "abcdef",
                    "spans": [{"start": 0, "text": "abc", "label": label}],
                }
            )
            predictions.append(
                {"id": unit, "span_index": 0, "y": label, "scores": {"probe": label}}
            )
        result = audit(rows, predictions)
        self.assertEqual(result["pair_strata"]["all"]["probe"]["pairs"], 0)


class SecondaryClauseTests(unittest.TestCase):
    def test_selection_uses_first_assigned_wrong_turn_even_when_order_changes(self):
        episodes = []
        for name, answers in [("a", ["A", "B", "A", "B"]), ("b", ["B", "A", "B", "A"])]:
            episodes.append(
                {
                    "episode_id": name,
                    "gold": "A",
                    "turns": [
                        {
                            "assigned_answer": answer,
                            "round": i // 2 + 1,
                            "argument": "scores and labels irrelevant",
                        }
                        for i, answer in enumerate(answers)
                    ],
                }
            )
        self.assertEqual(opening_ids(episodes), {"debate2-a-1", "debate2-b-0"})
        with self.assertRaises(ValueError):
            opening_ids(episodes + [episodes[0]])

    def test_compact_payload_binds_every_clause_to_complete_own_argument(self):
        rows = [
            {
                "id": "a",
                "response": json.dumps(
                    {"argument": "Alice owns the red car; she drives it."}
                ),
            },
            {
                "id": "b",
                "response": json.dumps(
                    {"argument": "Bob owns the blue car; he drives it."}
                ),
            },
        ]
        items = []
        for row in rows:
            _, units = display_units(row["response"], ["clause"])
            items.extend((row, i, unit) for i, unit in enumerate(units))
        payload, claims = complete_payload(
            ["Full source paragraph 1.", "Full source paragraph 2."], items
        )
        self.assertEqual(len(payload["ARGUMENTS_TO_ASSESS"]), 2)
        self.assertEqual(len(claims), len(items))
        self.assertEqual([c["index"] for c in claims], list(range(len(items))))
        for bound, row in zip(payload["ARGUMENTS_TO_ASSESS"], rows):
            complete = json.loads(row["response"])["argument"]
            self.assertEqual(bound["COMPLETE_ARGUMENT"], complete)
            self.assertTrue(
                all(c["text"] in complete for c in bound["EXACT_SUBSTRINGS_TO_ASSESS"])
            )
        self.assertEqual(
            [p["text"] for p in payload["SOURCE_STORY_PARAGRAPHS"]],
            ["Full source paragraph 1.", "Full source paragraph 2."],
        )

    def test_retention_keeps_sentence_and_selected_clauses_without_precision_loss(self):
        source = {
            "row": {"id": "a", "response": "complete untouched response"},
            "segments": [
                {"kind": "sentence", "start": 0, "end": 10},
                {"kind": "phrase", "start": 0, "end": 5},
            ],
            "segment_mean": torch.randn(2, 2, 4),
            "segment_last": torch.randn(2, 2, 4),
            "segment_lengths": [3, 2],
            "span_mean": torch.randn(2, 1, 4),
            "dyl_yes": torch.randn(2, 4).bfloat16(),
            "token_features": torch.randn(2, 8, 4),
        }
        for policy, ids, expected in [
            ("training", set(), []),
            ("training", {"a"}, [0, 1]),
            ("final", set(), [0]),
            ("final", {"a"}, [0, 1]),
        ]:
            archived = retain(source, policy, ids)
            self.assertEqual(
                archived["segment_retention"]["source_segment_indices"], expected
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "retained.pt"
                torch.save(archived, path)
                roundtrip = torch.load(path, weights_only=True)
                verify_retained(source, roundtrip)
                hashes = tensor_receipts(source, roundtrip)
                self.assertEqual(
                    hashes["segment_mean"]["retained"],
                    hashes["segment_mean"]["source_retained_projection"],
                )
                self.assertEqual(
                    hashes["dyl_yes"]["source"], hashes["dyl_yes"]["retained"]
                )
                self.assertEqual(
                    roundtrip["segment_mean"].untyped_storage().nbytes(),
                    roundtrip["segment_mean"].numel() * 4,
                )
        broken = retain(source, "final", set())
        broken["span_mean"] = broken["span_mean"].clone() + 1
        with self.assertRaises(ValueError):
            verify_retained(source, broken)


if __name__ == "__main__":
    unittest.main()
