"""Newest sessions first without reversing their source cursors."""
import json
import sqlite3
import unittest
from types import SimpleNamespace

from tess_memory.autobiography_extract import order_session_parts
import test_autobiography_extract as fixtures
from test_codex import write_rollout, message


class SessionOrderTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.db.row_factory = sqlite3.Row
        self.addCleanup(self.db.close)
        self.db.execute('CREATE TABLE autobio_sources(id,metadata,owner,agent,audience)')
        self.store = SimpleNamespace(db=self.db, principal=SimpleNamespace(owner='o', agent='a', audience='private'))

    def source(self, session, unit, time):
        md = dict(logical_source=session, unit=unit, roles=[dict(source_event_time=time)])
        self.db.execute('INSERT INTO autobio_sources VALUES(?,?,?,?,?)',
                        (f'{session}/unit/{unit}', json.dumps(md), 'o', 'a', 'private'))
        return md

    def part(self, session, unit, start, md):
        return dict(id=f'{session}-{unit}-{start}', source_id=f'{session}/unit/{unit}',
                    start=start, end=start+10, source_metadata=json.dumps(md))

    def test_session_tail_promotes_forward_cursor_not_just_newest_tokens(self):
        old = self.source('old', 0, '2026-10-17T00:00:00Z')
        a = self.source('active', 2, '2026-01-01T00:00:00Z')
        b = self.source('active', 10, '2026-10-18T00:00:00Z')
        rows = [self.part('old',0,0,old), self.part('active',10,0,b),
                self.part('active',2,10,a), self.part('active',2,0,a)]
        self.assertEqual([r['id'] for r in order_session_parts(self.store, rows)],
                         ['active-2-0','active-2-10','active-10-0','old-0-0'])

    def test_recomputed_on_next_run_even_if_newest_unit_already_completed(self):
        a = self.source('a',0,'2026-01-01T00:00:00Z')
        b = self.source('b',0,'2026-02-01T00:00:00Z')
        rows = [self.part('a',0,0,a),self.part('b',0,0,b)]
        self.assertEqual(order_session_parts(self.store,rows)[0]['id'],'b-0-0')
        self.source('a',1,'2026-03-01T00:00:00Z')
        self.assertEqual(order_session_parts(self.store,rows)[0]['id'],'a-0-0')

    def test_timestamp_ties_are_deterministic(self):
        a=self.source('a',0,None); b=self.source('b',0,'invalid')
        rows=[self.part('b',0,0,b),self.part('a',0,0,a)]
        self.assertEqual([r['id'] for r in order_session_parts(self.store,rows)],['a-0-0','b-0-0'])


class RefreshTests(unittest.TestCase):
    setUp = fixtures.ExtractionTests.setUp

    def test_refresh_is_not_truncated_at_128_sources(self):
        for i in range(128):
            write_rollout(self.home, sid=f'copy-{i}', items=[message('user', 'A new session.')])
        report=self.extract.prepare()
        self.assertEqual(report['scanned'],129)
        self.assertEqual(report['unscanned_this_pass'],0)
        self.assertFalse(report['errors'])

    def test_unlimited_prepare_keeps_single_pending_range(self):
        first=self.extract.prepare()
        self.assertEqual(len(first['parts']),1)
        self.assertEqual(self.extract.prepare()['parts'],first['parts'])
