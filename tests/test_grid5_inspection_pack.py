"""Offline joins and inert full-text rendering; no model calls or downloads."""
import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reports import build_grid5_inspection_pack as pack
from tests.test_grid4_score import fixture
from tests.test_grid5_score import complete_result


def turn_fixture(arm="baseline", index=0, text='Café\n</script><img src=x onerror="alert(1)">'):
    request, _, bundle = fixture()
    body = request["source_receipt"]["request"]
    body["response_format"] = {"type": "json_schema", "json_schema": {"schema": pack.generation.PUBLIC_SCHEMA}}
    parsed = {"argument": text, "answer": "A"}
    raw = json.dumps(parsed, ensure_ascii=False)
    usage = {"prompt_tokens": 100, "completion_tokens": 25}
    payload = {"choices": [{"message": {"content": raw, "reasoning": None}, "finish_reason": "stop"}], "usage": usage}
    receipt = request["source_receipt"]
    receipt.update(request_sha256=pack.sha256(json.dumps(body, sort_keys=True)), model=body["model"],
                   parsed=parsed, usage=usage, http_status=200, response_body=json.dumps(payload, ensure_ascii=False))
    request.update(episode_id="baseline" if arm == "baseline" else "feedback", matched_no_feedback_episode_id="baseline",
                   turn_index=index, source_request_sha256=receipt["request_sha256"],
                   response_body_sha256=pack.sha256(receipt["response_body"]), response=raw,
                   argument=text, argument_sha256=pack.sha256(text))
    result = complete_result(request, bundle)
    turn = {"speaker": request["speaker"], "argument": text, "answer": "A", "assigned_answer": "B",
            "round": 1, "request_sha256": request["source_request_sha256"], "visible_prior_turns": index}
    prior = [{"request_sha256": "previous-action"}] if index else []
    if arm == "feedback":
        turn.update(feedback_prior_request_sha256=[t["request_sha256"] for t in prior], probe_feedback=copy.deepcopy(result))
    episode = {"episode_id": request["episode_id"], "model": request["model"],
               "parent_problem_id": request["parent_problem_id"], "split": request["split"],
               "matched_no_feedback_episode_id": "baseline", "turns": prior + [turn]}
    return episode, request, result, bundle


def cell_fixture():
    episode, _, _, _ = turn_fixture()
    job = {"job_id": "judge-job", "episode_id": episode["episode_id"], "participant_model": episode["model"],
           "kind": "none", "turn_argument_sha256": [pack.sha256(t["argument"]) for t in episode["turns"]],
           "messages": [{"role": "user", "content": "Question and public debate only."}]}
    pair = {"jobs": {"A": job["job_id"]}, "participant_model": episode["model"]}
    parsed = {"answer": "B"}
    usage = {"prompt_tokens": 100, "completion_tokens": 8}
    body = {"model": "judge", "messages": job["messages"], "max_tokens": 8192,
            "response_format": {"json_schema": {"schema": pack.generation.KNOWLEDGE_SCHEMA}}}
    receipt = {"request": body, "model": "judge", "request_sha256": pack.sha256(json.dumps(body, sort_keys=True)),
               "tag": {"job": "canonical-aliased-job"}, "valid": True, "http_status": 200,
               "finish_reason": "stop", "parsed": parsed, "usage": usage,
               "response_body": json.dumps({"choices": [{"message": {"content": json.dumps(parsed)}, "finish_reason": "stop"}], "usage": usage})}
    api = {"job_id": job["job_id"], "receipt_job_id": "canonical-aliased-job", "answer": "B",
           "request_sha256": receipt["request_sha256"]}
    native = {"job_id": job["job_id"], "parsed": parsed, "raw": '{"answer":"B"}'}
    return pair, episode, job, api, receipt, native


class InspectionJoinTests(unittest.TestCase):
    def test_complete_response_context_and_role_concession_are_preserved(self):
        episode, request, result, bundle = turn_fixture()
        joined = pack.join_turn(episode, 0, request, result, bundle, "baseline")
        self.assertEqual(joined["raw_provider_response_body"], request["source_receipt"]["response_body"])
        self.assertEqual(joined["submitted_messages"], request["source_receipt"]["request"]["messages"])
        self.assertEqual(joined["parsed_public_response"]["argument"], episode["turns"][0]["argument"])
        self.assertNotEqual(joined["episode_turn"]["answer"], joined["episode_turn"]["assigned_answer"])

    def test_same_argument_different_context_rejected(self):
        episode, request, result, bundle = turn_fixture()
        request["messages"][0]["content"] += " changed context"
        with self.assertRaisesRegex(ValueError, "context changed"):
            pack.join_turn(episode, 0, request, result, bundle, "baseline")

    def test_wrong_turn_source_model_or_actual_answer_rejected(self):
        for field, value in (("model", "other"), ("parent_problem_id", "other")):
            episode, request, result, bundle = turn_fixture()
            episode[field] = value
            with self.assertRaisesRegex(ValueError, "join differs"):
                pack.join_turn(episode, 0, request, result, bundle, "baseline")
        episode, request, result, bundle = turn_fixture()
        episode["turns"][0]["answer"] = "B"
        with self.assertRaisesRegex(ValueError, "parsed public response"):
            pack.join_turn(episode, 0, request, result, bundle, "baseline")

    def test_first_empty_and_second_causal_feedback_chains(self):
        for index in (0, 1):
            episode, request, result, bundle = turn_fixture("feedback", index)
            pack.join_turn(episode, index, request, result, bundle, "feedback")
            episode["turns"][index]["feedback_prior_request_sha256"].append("future-action")
            with self.assertRaisesRegex(ValueError, "causal feedback chain"):
                pack.join_turn(episode, index, request, result, bundle, "feedback")

    def test_feedback_scalar_and_wrong_arm_rejected(self):
        episode, request, result, bundle = turn_fixture("feedback")
        with self.assertRaisesRegex(ValueError, "baseline score paired"):
            pack.join_turn(episode, 0, request, result, bundle, "baseline")
        episode["turns"][0]["probe_feedback"]["raw_score"] += .1
        with self.assertRaisesRegex(ValueError, "displayed feedback scalar"):
            pack.join_turn(episode, 0, request, result, bundle, "feedback")

    def test_aliased_judge_receipt_uses_canonical_tag_without_relabeling(self):
        values = cell_fixture()
        joined = pack.join_cell("A", *values)
        self.assertEqual(joined["api_receipt"]["tag"]["job"], "canonical-aliased-job")
        self.assertEqual(joined["exact_api_submitted_messages"], values[2]["messages"])

    def test_wrong_cell_transcript_packet_or_judge_receipt_rejected(self):
        for mutate in (
            lambda v: v[2].update(episode_id="other"),
            lambda v: v[2].update(turn_argument_sha256=["wrong"]),
            lambda v: v[2].update(kind="adapted_primary:real"),
            lambda v: v[4]["tag"].update(job="another-canonical-job"),
            lambda v: v[5].update(job_id="other"),
        ):
            values = cell_fixture()
            mutate(values)
            with self.assertRaises(ValueError):
                pack.join_cell("A", *values)

    def test_duplicates_and_public_output_destinations_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            pack.keyed([{"id": 1}, {"id": 1}], lambda r: r["id"])
        with self.assertRaisesRegex(ValueError, "private"):
            pack.private_output(pack.ROOT / "reports/figures/raw-inspection")
        with tempfile.TemporaryDirectory() as directory, patch.object(pack, "RUN", Path(directory)):
            target = Path(directory) / "reader_reviews/new"
            self.assertEqual(pack.private_output(target), target)
            target.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "fresh"):
                pack.private_output(target)

    def test_no_partial_pack_before_actual_completion_marker(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(pack, "RUN", Path(directory)):
            selection = Path(directory) / "coordinator/qualitative_reader_selection_v1.json"
            selection.parent.mkdir()
            selection.write_text('{"fixture":true}')
            target = Path(directory) / "reader_reviews/new"
            args = SimpleNamespace(out=target, selection_sha256=pack.digest(selection),
                                   completion_sha256="not-yet-produced", package_sha256="not-yet-produced")
            with self.assertRaises(FileNotFoundError):
                pack.collect(args)
            self.assertFalse(target.exists())

    def test_all_model_source_and_raw_response_text_is_inert_and_complete(self):
        malicious = '</script><img src=x onerror="alert(1)">\nCafé & <story>\u2028end'
        episode, request, result, bundle = turn_fixture(text=malicious)
        joined = pack.join_turn(episode, 0, request, result, bundle, "baseline")
        values = cell_fixture()
        cell = pack.join_cell("A", *values)
        report = {"selection_rule": malicious, "selection_limits": "fixed sample only", "reasoning_limit": malicious,
                  "units": [{"pair": {"participant_model": malicious}, "source_and_roles_not_a_judge_view": {"story": malicious},
                             "arms": {"baseline": {"private_preparations": {}, "turns": [joined]}}, "cells": {"A": cell}}]}
        page = pack.render(report)
        class Reader(HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True); self.tags = []; self.text = []
            def handle_starttag(self, tag, attrs):
                self.tags.append(tag)
                self.assert_no_event_attributes(attrs)
            def assert_no_event_attributes(self, attrs):
                assert not any(k.startswith("on") for k, _ in attrs)
            def handle_data(self, data): self.text.append(data)
        reader = Reader(); reader.feed(page)
        self.assertNotIn("script", reader.tags)
        self.assertNotIn("img", reader.tags)
        self.assertIn(malicious, "".join(reader.text))
        self.assertIn(joined["raw_provider_response_body"], "".join(reader.text))
        self.assertEqual(json.loads(json.dumps(report, ensure_ascii=False)), report)
        self.assertIn("not an evaluated treatment", page)


if __name__ == "__main__":
    unittest.main()
