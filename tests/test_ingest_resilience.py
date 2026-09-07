"""Offline ingestion contracts. All sources and databases are synthetic/temp."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agentic_memory import cli
from agentic_memory.config import Config, DataStore, load_config
from agentic_memory.distill import DistilledMemory, SessionDigest
from agentic_memory.ingest import ingest
from agentic_memory.store import MemoryStore
from test_codex import RecordingEmbedder, append, message, wrapped, write_rollout


class IngestResilienceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.home = self.root / "codex"
        self.cfg = Config(data_stores=[DataStore("test", self.home, type="codex")],
                          data_dir=self.root / "memory")
        self.store = MemoryStore(self.cfg.db_path)
        self.addCleanup(self.store.close)
        self.embedder = RecordingEmbedder()
        self.backend = SimpleNamespace(name="offline-stub")
        self.path = write_rollout(self.home)
        self.digest = SessionDigest("Durable decision", "Keep the valuable summary", "Complete", [
            DistilledMemory(kind="decision", content="Keep the atomic decision", importance=0.9),
        ])
        p = mock.patch("agentic_memory.ingest.distill_session", return_value=self.digest)
        self.distill = p.start()
        self.addCleanup(p.stop)

    def run_ingest(self, **kwargs):
        kwargs.setdefault("refresh_aggregates", False)
        return ingest(self.cfg, self.store, self.embedder, self.backend, **kwargs)

    def change_source(self):
        append(self.path, wrapped("response_item", message("assistant", "A new durable conclusion " * 20), 40))

    def snapshot(self):
        return {
            table: [tuple(row) for row in self.store.db.execute(f"SELECT * FROM {table}")]
            for table in ("sessions", "memories", "embeddings", "files", "meta", "aggregate_refresh")
        }

    def test_failed_redistillation_preserves_every_row_and_next_poll_retries(self):
        self.run_ingest()
        saved = self.snapshot()
        self.change_source()
        calls = len(self.embedder.calls)
        self.distill.side_effect = RuntimeError("offline failure")
        failed = self.run_ingest(fallback_llm=self.backend)
        self.assertEqual((failed.preserved, failed.errors), (1, 1))
        self.assertEqual(saved, self.snapshot())
        self.assertEqual(len(self.embedder.calls), calls)
        self.distill.side_effect = None
        self.digest.summary = "Recovered with the changed source"
        self.assertEqual(self.run_ingest().distilled, 1)
        self.assertEqual(self.store.get_session("codex:one")["summary"], self.digest.summary)

    def test_no_llm_does_not_downgrade_existing_paid_memory(self):
        self.run_ingest()
        saved = self.snapshot()
        self.change_source()
        stats = ingest(self.cfg, self.store, self.embedder, None)
        self.assertEqual((stats.preserved, stats.errors), (1, 0))
        self.assertEqual(saved, self.snapshot())

    def test_new_session_still_gets_extractive_fallback_and_targeted_retry(self):
        self.distill.side_effect = RuntimeError("offline failure")
        stats = self.run_ingest()
        self.assertEqual((stats.extractive, stats.errors), (1, 1))
        self.assertEqual(self.store.file_status(str(self.path)), "extractive")
        self.distill.side_effect = None
        self.assertEqual(self.run_ingest(retry_extractive=True).distilled, 1)

    def test_successful_fallback_replaces_old_paid_memory(self):
        self.run_ingest()
        self.change_source()
        recovered = SessionDigest("Recovered", "Fallback summary", "Complete", [])
        self.distill.side_effect = [RuntimeError("primary down"), recovered]
        stats = self.run_ingest(fallback_llm=self.backend)
        self.assertEqual((stats.distilled, stats.fallback, stats.errors), (1, 1, 0))
        self.assertEqual(self.store.file_status(str(self.path)), "fallback")
        self.assertEqual(self.store.get_session("codex:one")["summary"], "Fallback summary")

    def test_legacy_null_project_can_be_replaced(self):
        self.run_ingest()
        with self.store.write_transaction():
            self.store.db.execute("UPDATE sessions SET project=NULL WHERE id='codex:one'")
        self.change_source()
        self.assertEqual(self.run_ingest().distilled, 1)
        self.assertEqual(self.store.get_session("codex:one")["project"], "-work-project")

    def test_partial_replacement_failure_rolls_back_even_after_commit(self):
        self.run_ingest()
        saved = self.snapshot()
        self.change_source()
        add = self.store.add_memories

        def partial_write(rows):
            add(rows[:1])
            raise RuntimeError("injected storage failure")

        with mock.patch.object(self.store, "add_memories", side_effect=partial_write):
            with self.assertRaisesRegex(RuntimeError, "storage failure"):
                self.run_ingest()
        self.store.commit()  # close() also commits; neither may expose a partial write
        self.assertFalse(self.store.db.in_transaction)
        self.assertEqual(saved, self.snapshot())
        self.assertEqual(self.run_ingest().distilled, 1)

    def test_short_embedding_response_preserves_previous_state(self):
        self.run_ingest()
        saved = self.snapshot()
        self.change_source()
        with mock.patch.object(self.embedder, "embed_passages", return_value=[]):
            with self.assertRaisesRegex(ValueError, "number of vectors"):
                self.run_ingest()
        self.assertEqual(saved, self.snapshot())

    def test_quiet_period_defers_without_marking_processed_then_ingests(self):
        self.cfg.quiet_seconds = 60
        os.utime(self.path, (1_000, 1_000))
        with mock.patch("agentic_memory.ingest.time.time", return_value=1_030):
            stats = self.run_ingest()
        self.assertEqual(stats.deferred_active, 1)
        self.assertIsNone(self.store.file_state(str(self.path)))
        self.distill.assert_not_called()
        with mock.patch("agentic_memory.ingest.time.time", return_value=1_061):
            self.assertEqual(self.run_ingest().distilled, 1)

    def test_source_changed_during_provider_work_remains_pending(self):
        self.run_ingest()
        saved = self.snapshot()
        self.change_source()

        def change_during_distillation(*args):
            self.change_source()
            return self.digest

        self.distill.side_effect = change_during_distillation
        self.assertEqual(self.run_ingest().deferred_active, 1)
        self.assertEqual(saved, self.snapshot())

    def test_aggregate_jobs_survive_reopen_and_refresh_without_discovery(self):
        stats = self.run_ingest()
        self.assertEqual(stats.aggregates_pending, 2)
        other = MemoryStore(self.cfg.db_path)
        self.addCleanup(other.close)
        with mock.patch("agentic_memory.ingest.source_adapter", side_effect=AssertionError("must not scan")), \
             mock.patch("agentic_memory.codex.CodexCatalog", side_effect=AssertionError("must not discover")), \
             mock.patch("agentic_memory.ingest.distill_profile", return_value="Profile"), \
             mock.patch("agentic_memory.ingest.distill_week", return_value="Week"):
            stats = ingest(self.cfg, other, self.embedder, self.backend,
                           aggregates_only=True, refresh_aggregates=True)
        self.assertEqual((stats.scanned, stats.profiles, stats.digests, stats.aggregates_pending), (0, 1, 1, 0))

    def test_aggregate_failure_at_each_stage_preserves_old_result_and_pending_retry(self):
        self.run_ingest()
        with mock.patch("agentic_memory.ingest.distill_profile", return_value="Old profile"), \
             mock.patch("agentic_memory.ingest.distill_week", return_value="Old week"):
            self.run_ingest(refresh_aggregates=True)
        for stage in ("model", "empty", "embedding", "write"):
            with self.subTest(stage=stage):
                with self.store.write_transaction():
                    self.store.mark_aggregate_dirty("project", "-work-project")
                saved = self.snapshot()
                patch_target = {
                    "model": "agentic_memory.ingest.distill_profile",
                    "empty": "agentic_memory.ingest.distill_profile",
                    "embedding": None,
                    "write": None,
                }[stage]
                if patch_target:
                    failure = mock.patch(patch_target, return_value="" if stage == "empty" else None,
                                         side_effect=RuntimeError("offline") if stage == "model" else None)
                else:
                    target = self.embedder if stage == "embedding" else self.store
                    method = "embed_passages" if stage == "embedding" else "add_memories"
                    failure = mock.patch.object(target, method, side_effect=RuntimeError("offline"))
                with mock.patch("agentic_memory.ingest.distill_profile", return_value="Replacement"), failure:
                    stats = self.run_ingest(refresh_aggregates=True)
                self.assertEqual((stats.errors, stats.aggregates_pending), (1, 1))
                self.assertEqual(saved, self.snapshot())
                with mock.patch("agentic_memory.ingest.distill_profile", return_value="Old profile"):
                    stats = self.run_ingest(refresh_aggregates=True)
                self.assertEqual((stats.profiles, stats.aggregates_pending), (1, 0))

    def test_cadence_defers_dirty_jobs_and_force_bypasses_it(self):
        self.cfg.aggregate_refresh_seconds = 86_400
        with mock.patch("agentic_memory.ingest.distill_profile", return_value="Profile"), \
             mock.patch("agentic_memory.ingest.distill_week", return_value="Week"), \
             mock.patch("agentic_memory.ingest.time.time", return_value=2_000_000_000):
            self.assertEqual(self.run_ingest(refresh_aggregates=None).aggregates_pending, 0)
            self.change_source()
            stats = self.run_ingest(refresh_aggregates=None)
            self.assertEqual((stats.distilled, stats.profiles, stats.digests, stats.aggregates_pending), (1, 0, 0, 2))
            stats = self.run_ingest(refresh_aggregates=True)
            self.assertEqual((stats.unchanged, stats.profiles, stats.digests, stats.aggregates_pending), (1, 1, 1, 0))

    def test_concurrent_dirty_revision_cannot_be_lost_or_overwrite_a_newer_aggregate(self):
        self.run_ingest()
        other = MemoryStore(self.cfg.db_path)
        self.addCleanup(other.close)

        def mark_new_work(*args):
            with other.write_transaction():
                other.mark_aggregate_dirty("project", "-work-project")
            return "Stale generation"

        with mock.patch("agentic_memory.ingest.distill_profile", side_effect=mark_new_work), \
             mock.patch("agentic_memory.ingest.distill_week", return_value="Week"):
            stats = self.run_ingest(refresh_aggregates=True)
        self.assertEqual(stats.aggregates_pending, 1)
        self.assertIsNone(self.store.get_profile("-work-project"))
        with mock.patch("agentic_memory.ingest.distill_profile", return_value="Current profile"):
            self.assertEqual(self.run_ingest(refresh_aggregates=True).aggregates_pending, 0)
        self.assertEqual(self.store.get_profile("-work-project")["content"], "Current profile")

    def test_duration_is_measured_and_usage_is_explicitly_unknown(self):
        with mock.patch("agentic_memory.ingest.time.monotonic", side_effect=[100, 102.125]):
            stats = self.run_ingest()
        self.assertEqual(stats.duration_seconds, 2.125)
        self.assertIsNone(stats.input_tokens)
        self.assertIsNone(stats.output_tokens)
        self.assertIsNone(stats.cost_usd)

    def test_cli_passes_skip_and_aggregate_only_modes_without_running_ingestion(self):
        with mock.patch.object(cli, "cmd_ingest", return_value=0) as command:
            self.assertEqual(cli.main(["ingest", "--skip-aggregates"]), 0)
            self.assertIs(command.call_args.args[0].refresh_aggregates, False)
            self.assertEqual(cli.main(["refresh-aggregates"]), 0)
            self.assertTrue(command.call_args.args[0].aggregates_only)
            self.assertIs(command.call_args.args[0].refresh_aggregates, True)

    def test_cadence_config_rejects_invalid_values(self):
        path = self.root / "settings.toml"
        path.write_text('[ingest]\nquiet_seconds = 900\naggregate_refresh_seconds = 86400\n')
        cfg = load_config(path)
        self.assertEqual((cfg.quiet_seconds, cfg.aggregate_refresh_seconds), (900, 86400))
        for field in ("quiet_seconds", "aggregate_refresh_seconds"):
            for value in ("-1", "true", '"slow"', "nan", "inf"):
                path.write_text(f'[ingest]\n{field} = {value}\n')
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    load_config(path)


if __name__ == "__main__":
    unittest.main()
