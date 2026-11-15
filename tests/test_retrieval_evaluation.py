"""Small deterministic evaluation of retrieval contracts, not model quality.

Synthetic decision/plan fixtures, a fixed clock, deterministic vectors and
scripted agent actions cover current-vs-superseded evidence, live plan status,
absent evidence and scope isolation. No history files or providers are used.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np

from agentic_memory import cli, plan, retrieval, server
from agentic_memory.config import Config
from agentic_memory.search_agent import agentic_search
from agentic_memory.store import MemoryRow, MemoryStore


class FixedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 5, tzinfo=timezone.utc)


class OfflineEmbedder:
    descriptor = "test:retrieval-evaluation:2"

    def embed_passages(self, texts):
        return np.tile(np.asarray([1, 0], dtype=np.float32), (len(texts), 1))

    def embed_query(self, query):
        return np.asarray([1, 0], dtype=np.float32)


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = MemoryStore(Path(tmp.name) / "memory.db")
        self.addCleanup(self.store.close)
        self.embedder = OfflineEmbedder()
        self.cfg = Config()
        for patcher in (mock.patch.object(retrieval, "CONFIG", self.cfg),
                        mock.patch.object(retrieval, "datetime", FixedClock)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def add(self, content, *, project="atlas", level="memory", kind="decision",
            ts="2026-09-01", vector=True, importance=0.8):
        ids = self.store.add_memories([MemoryRow(
            level, kind, content, project=project, ts=ts, importance=importance,
            vector=self.embedder.embed_query("") if vector else None,
        )])
        self.store.commit()
        return ids[0]

    def test_current_decision_outranks_equally_matching_superseded_history(self):
        old = self.add("Atlas cache decision: choose Redis; superseded by newer decision.", ts="2026-01-01")
        current = self.add("Atlas cache decision: choose Valkey; replaces the earlier decision.")
        hits = retrieval.search(self.store, self.embedder, "Atlas cache decision", project="atlas")
        self.assertEqual([hit["id"] for hit in hits], [current, old])

    def test_completed_and_pending_plans_retrieve_live_status_without_stale_mirror(self):
        with mock.patch("agentic_memory.plan._now", return_value="2026-09-01T00:00:00Z"):
            done = plan.add_item(self.store, self.embedder, project="atlas", kind="task",
                                 title="Ship cache migration", body="Upgrade the cache service.")
            pending = plan.add_item(self.store, self.embedder, project="atlas", kind="task",
                                    title="Ship cache migration", body="Upgrade the cache service.")
            plan.update_item(self.store, self.embedder, done["id"], status="complete")
        hits = retrieval.search(self.store, self.embedder, "cache migration", project="atlas", levels=("plan",))
        self.assertIn(f"#{pending['id']} | planned", hits[0]["content"])
        self.assertIn(f"#{done['id']} | complete", hits[1]["content"])
        self.assertEqual(len(hits), 2)
        stale = self.store.db.execute("SELECT content FROM memories WHERE plan_id=?", (done["id"],)).fetchall()
        self.assertEqual(len(stale), 1)
        self.assertNotIn("| planned |", stale[0]["content"])

    def test_unrelated_projects_cannot_crowd_out_keyword_only_scoped_evidence(self):
        for _ in range(125):
            self.add("needle", project="elsewhere", vector=False)
        wanted = self.add("needle with a longer descriptive context", vector=False)
        hits = retrieval.search(self.store, self.embedder, "needle", project="atlas", candidates=3)
        self.assertEqual([h["id"] for h in hits], [wanted])

    def test_all_keyword_filters_apply_before_top_n_and_existing_api_still_works(self):
        for fields in ({"project": "elsewhere"}, {"level": "chunk"}, {"kind": "fact"},
                       {"ts": "2026-01-01"}, {"ts": "2027-01-01"}):
            self.add("needle", vector=False, **fields)
        wanted = self.add("needle with enough extra words to rank lower globally", vector=False)
        self.assertTrue(self.store.fts_search("needle", 1))
        matches = self.store.fts_search("needle", 1, project="atlas", levels=("memory",),
                                        kinds=("decision",), after="2026-08-01", before="2026-10-01")
        self.assertEqual(list(matches), [wanted])

    def test_empty_scope_short_circuits_static_and_agent_calls(self):
        self.add("Other project exists", project="elsewhere")
        with mock.patch.object(self.embedder, "embed_query", side_effect=AssertionError("embedding called")), \
             mock.patch.object(self.store, "assert_embedder", side_effect=AssertionError("embedder guard called")):
            self.assertEqual(retrieval.search(self.store, self.embedder, "anything", project="missing"), [])
            llm = mock.Mock()
            hits, trace = agentic_search(self.cfg, self.store, self.embedder, llm, "anything", project="missing")
            self.assertEqual((hits, trace.rounds, trace.queries), ([], 0, []))
            llm.complete.assert_not_called()

    def test_empty_first_query_does_not_stop_agent_in_a_nonempty_scope(self):
        wanted = self.add("Valkey cache migration result", vector=False)
        llm = mock.Mock()
        llm.complete.side_effect = [json.dumps({"action": "search", "queries": ["Valkey"]}),
                                    json.dumps({"action": "finish", "keep": [wanted]})]
        hits, trace = agentic_search(self.cfg, self.store, self.embedder, llm, "unmatchedquux", project="atlas")
        self.assertEqual([h["id"] for h in hits], [wanted])
        self.assertEqual(trace.rounds, 2)

    def test_absent_evidence_can_abstain_despite_nearest_vector_candidates(self):
        self.add("Atlas has a cache migration decision.")
        llm = mock.Mock()
        llm.complete.return_value = json.dumps({"action": "finish", "keep": [],
                                               "note": "No evidence about the requested billing policy."})
        hits, trace = agentic_search(self.cfg, self.store, self.embedder, llm, "Atlas billing policy", project="atlas")
        self.assertEqual(hits, [])
        self.assertEqual(trace.rounds, 1)
        self.assertIn("No evidence", trace.note)

    def test_malformed_selection_and_provider_failure_still_fall_back_to_seed(self):
        wanted = self.add("Atlas cache migration decision")
        for action in ({"action": "finish"}, {"action": "finish", "keep": "bad"},
                       {"action": "finish", "keep": [999999]}):
            llm = mock.Mock()
            llm.complete.return_value = json.dumps(action)
            hits, _ = agentic_search(self.cfg, self.store, self.embedder, llm, "cache", project="atlas")
            self.assertEqual([h["id"] for h in hits], [wanted])
        llm.complete.side_effect = RuntimeError("offline")
        hits, _ = agentic_search(self.cfg, self.store, self.embedder, llm, "cache", project="atlas")
        self.assertEqual([h["id"] for h in hits], [wanted])

    def test_full_session_expansion_cannot_escape_constraints_or_invent_session_ids(self):
        self.add("Atlas cache migration decision")
        llm = mock.Mock()
        llm.complete.return_value = json.dumps({"action": "get_session", "session_ids": ["outside"]})
        with mock.patch("agentic_memory.search_agent._session_evidence", side_effect=AssertionError("expanded")):
            for filters in ({"project": "atlas"}, {"project": "atlas", "level": "memory"}):
                hits, trace = agentic_search(self.cfg, self.store, self.embedder, llm, "cache", **filters)
                self.assertEqual(len(hits), 1)
                self.assertEqual(trace.sessions, [])

    def test_invalid_session_reference_gets_bounded_feedback_before_valid_finish(self):
        wanted = self.add('Atlas cache migration decision')
        llm = mock.Mock()
        llm.complete.side_effect = [json.dumps({'action': 'get_session', 'session_ids': ['invented']}),
                                    json.dumps({'action': 'finish', 'keep': [wanted]})]
        with mock.patch('agentic_memory.search_agent._session_evidence', side_effect=AssertionError('must not expand invented id')):
            hits, trace = agentic_search(self.cfg, self.store, self.embedder, llm, 'cache', project='atlas')
        self.assertEqual([h['id'] for h in hits], [wanted])
        self.assertEqual(trace.invalid_actions, 1)
        self.assertFalse(trace.fallback_used)
        self.assertTrue(trace.model_finished)
        self.assertIn('PREVIOUS ACTION FEEDBACK', llm.complete.call_args.kwargs['prompt'])

    def test_mcp_empty_scope_does_not_construct_providers(self):
        self.add("Unrelated corpus", project="elsewhere")
        with mock.patch.object(server, "_get_store", return_value=self.store), \
             mock.patch.object(server, "_get_embedder", side_effect=AssertionError("embedder initialized")), \
             mock.patch.object(server, "_get_search_llm", side_effect=AssertionError("LLM initialized")):
            self.assertIn("No matching memories", server.search_memory("anything", project="missing"))

    def test_mcp_static_mode_avoids_retrieval_llm_and_agentic_mode_can_opt_in(self):
        self.add("Atlas cache migration decision")
        with mock.patch.object(server, "_get_store", return_value=self.store), \
             mock.patch.object(server, "_get_embedder", return_value=self.embedder), \
             mock.patch.object(server, "_get_search_llm", side_effect=AssertionError("LLM initialized")):
            self.assertIn("cache migration", server.search_memory("cache", project="atlas", mode="static"))
        with mock.patch.object(server, "CONFIG", Config(search_agent=False)), \
             mock.patch.object(server, "_search_llm", None), \
             mock.patch("agentic_memory.llm.AgentCoreBackend.for_role", return_value="backend") as build:
            self.assertIsNone(server._get_search_llm())
            build.assert_not_called()
            self.assertEqual(server._get_search_llm(force=True), "backend")
            build.assert_called_once()

    def test_cli_empty_scope_avoids_provider_initialization(self):
        self.add("Unrelated corpus", project="elsewhere")
        with mock.patch.object(cli, "CONFIG", self.cfg), \
             mock.patch("agentic_memory.store.MemoryStore", return_value=self.store), \
             mock.patch.object(self.store, "close"), \
             mock.patch("agentic_memory.embed.make_embedder", side_effect=AssertionError("embedder initialized")), \
             mock.patch("agentic_memory.llm.AgentCoreBackend.for_role", side_effect=AssertionError("LLM initialized")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            args = argparse.Namespace(project="missing", level=None, limit=None, query="anything", static=False)
            self.assertEqual(cli.cmd_search(args), 0)
        self.assertIn("no results", output.getvalue())


if __name__ == "__main__":
    unittest.main()
