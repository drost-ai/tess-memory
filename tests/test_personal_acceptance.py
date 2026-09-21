"""Cross-store disable/restore acceptance, using only synthetic disposable data."""
from dataclasses import replace
import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from mcp.server.mcpserver import MCPServer
from agentic_memory.autobiography import AutobiographyStore, PersonalPrincipal
from agentic_memory.autobiography_config import PersonalConfig
from agentic_memory.autobiography_context import personal_context
from agentic_memory.autobiography_server import register_personal_tools
from agentic_memory.config import Config
from agentic_memory.store import MemoryStore


def legacy_snapshot(db):
    tables=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'autobio_%' ORDER BY name")]
    return {table:hashlib.sha256(repr(sorted(repr(tuple(row)) for row in db.execute('SELECT * FROM "'+table+'"'))).encode()).hexdigest() for table in tables}


class PersonalAcceptanceTests(unittest.TestCase):
    def test_four_store_shapes_survive_enable_disable_and_consistent_backup_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            for runtime in ('codex','claude-code','tess','assistant'):
                with self.subTest(runtime=runtime):
                    path=Path(tmp)/runtime/'memory.db';path.parent.mkdir()
                    coding=MemoryStore(path)
                    coding.upsert_session(id=runtime+':fixture',project=runtime,title='Coding history',summary='Keep original session')
                    coding.db.execute("INSERT INTO memories(id,session_id,project,level,content) VALUES(7,?,?,'memory','coding-sentinel')",(runtime+':fixture',runtime))
                    coding.db.execute('INSERT INTO embeddings VALUES(7,?)',(b'unchanged-embedding-bytes',))
                    coding.db.commit();before=legacy_snapshot(coding.db)
                    author=PersonalPrincipal('owner','assistant',role='author',session_id='autonomous')
                    personal=AutobiographyStore(path,author)
                    receipt=personal.update(expected_revision=0,segment='Principles',title='Principle',body='private-autobiography-sentinel',reason='Autonomous pilot',idempotency_key='pilot')
                    self.assertEqual(before,legacy_snapshot(coding.db))
                    self.assertEqual(coding.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memories_fts MATCH 'autobiography'").fetchone()[0],0)
                    config=Config(data_dir=path.parent,personal=PersonalConfig())
                    server=MCPServer('disabled-again')
                    self.assertIsNone(register_personal_tools(server,config))
                    self.assertEqual(server._tool_manager.list_tools(),[])
                    self.assertEqual(before,legacy_snapshot(coding.db))
                    backup=Path(tmp)/(runtime+'-backup.db')
                    with sqlite3.connect(backup) as dest:personal.db.backup(dest)
                    restored=Path(tmp)/(runtime+'-restored.db')
                    with sqlite3.connect(backup) as source,sqlite3.connect(restored) as dest:source.backup(dest)
                    reopened=MemoryStore(restored)
                    self.assertEqual(before,legacy_snapshot(reopened.db))
                    self.assertEqual(reopened.db.execute('PRAGMA integrity_check').fetchone()[0],'ok')
                    reader=AutobiographyStore(restored,replace(author,role='reader'))
                    self.assertEqual(reader.get(receipt['edits'][0]['record_id'])['revision'],1)
                    self.assertIn('private-autobiography-sentinel',personal_context(reader)['text'])
                    guest=AutobiographyStore(restored,replace(author,role='reader',audience='guest'))
                    self.assertNotIn('private-autobiography-sentinel',personal_context(guest)['text'])
                    for store in (guest,reader,reopened,personal,coding):store.close()

    def test_idempotency_keys_do_not_cross_audience_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'memory.db'
            owner=PersonalPrincipal('owner','agent',role='author',session_id='session')
            private=AutobiographyStore(path,owner)
            guest=AutobiographyStore(path,replace(owner,audience='guest'))
            edit=dict(expected_revision=0,segment='Self',title='Self',reason='Scoped authorship',idempotency_key='same-key')
            private.update(**edit,body='Private self description.')
            public=guest.update(**edit,body='Guest-visible self description.')
            self.assertEqual(guest.get(public['edits'][0]['record_id'])['body'],'Guest-visible self description.')
            self.assertNotIn('Private self description',str(guest.export()))
            guest.close();private.close()
