"""Personal memory per person: in a multi-audience store each
person's records are theirs, the agent's Self/Principles are shared, the
owner reads all; single-audience stores behave exactly as before."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tess_memory.autobiography import AutobiographyStore, PersonalError, PersonalPrincipal
from tess_memory.autobiography_config import PersonalConfig, session_audience
from tess_memory.autobiography_context import personal_context
from tess_memory.personal_operations import PersonalOperations

CONFIG = dict(enabled=True, owner="sam", agent="tess", audience="sam", role="author", session_id="s-1", multi_audience=True)


class AudienceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "memory.db"
        self.cfg = PersonalConfig(**CONFIG)

    def store(self, audience=None, cfg=None):
        store = AutobiographyStore(self.path, (cfg or self.cfg).principal(audience=audience))
        self.addCleanup(store.close)
        return store

    @staticmethod
    def write(store, segment, title, body, key, subject=None):
        edit = {"expected_revision": 0, "segment": segment, "title": title, "body": body,
                "reason": "test authorship", "idempotency_key": key}
        if subject:
            edit["subject"] = subject
        return store.update(**edit)["edits"][0]["record_id"]

    def test_each_person_reads_their_own_plus_the_shared_self_and_the_owner_reads_all(self):
        owner = self.store()
        self_id = self.write(owner, "Identity", "Who Tess is", "Tess is curious and careful.", "self")
        self.write(owner, "People", "Sam's sister", "Sam's sister lives in Kandy.", "sam-people", subject="sister")
        ana = self.store("ana")
        self.write(ana, "People", "Ana's dog", "Ana has a dog named Pixel.", "ana-people", subject="pixel")

        ana_titles = {r["title"] for r in ana.list_records()["records"]}
        self.assertEqual(ana_titles, {"Who Tess is", "Ana's dog"})
        self.assertEqual({r["title"] for r in owner.list_records()["records"]}, {"Who Tess is", "Sam's sister", "Ana's dog"})
        with self.assertRaises(PersonalError):  # the shared self is readable, not hers to edit
            ana.update(record_id=self_id, expected_revision=1, title="x", body="Rewritten.", reason="no", idempotency_key="steal")

        self.assertEqual([r["title"] for r in ana.search("Kandy")["records"]], [])
        self.assertEqual([r["title"] for r in ana.search("curious")["records"]], ["Who Tess is"])
        self.assertEqual([r["title"] for r in owner.search("Pixel")["records"]], ["Ana's dog"])

        context = personal_context(ana, budget=6000)
        self.assertIn("Tess is curious", context["text"])
        self.assertNotIn("Kandy", context["text"])
        self.assertNotIn("Pixel", personal_context(owner, budget=6000, subjects=["pixel"])["text"])  # owner context stays the owner's

    def test_another_audience_needs_multi_audience_and_single_stores_are_unchanged(self):
        single = PersonalConfig(**{**CONFIG, "multi_audience": False})
        with self.assertRaises(PersonalError):
            single.principal(audience="ana")
        self.assertEqual(single.principal().reads, "")
        self.assertEqual(self.cfg.principal().reads, "*")
        self.assertEqual(self.cfg.principal(audience="ana").reads, "sam")
        with self.assertRaises(ValueError):
            PersonalConfig.parse({**CONFIG, "multi_audience": "yes"})

    def test_sessions_belong_to_the_audience_their_metadata_names(self):
        self.assertEqual(session_audience({"metadata": {"personalAudience": "ana"}}, self.cfg), "ana")
        self.assertEqual(session_audience({"metadata": {}}, self.cfg), "sam")
        self.assertEqual(session_audience({"metadata": {"personalAudience": "bad id!"}}, self.cfg), "sam")

    def test_each_audience_has_its_own_job_schedule(self):
        for audience, expected in ((None, "extractor"), ("ana", "extractor@ana")):
            operations = PersonalOperations.__new__(PersonalOperations)
            operations.store = SimpleNamespace(principal=self.cfg.principal(audience=audience))
            operations.config = SimpleNamespace(personal=self.cfg)
            self.assertEqual(operations._schedule_key("extractor"), expected)

    def test_a_reads_value_is_validated(self):
        with self.assertRaises(PersonalError):
            PersonalPrincipal("sam", "tess", reads="not valid!")


if __name__ == "__main__":
    unittest.main()
