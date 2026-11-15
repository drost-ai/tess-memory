from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from mcp.server.mcpserver import MCPServer
from agentic_memory.autobiography import AutobiographyStore,PersonalPrincipal,PersonalError
from agentic_memory.autobiography_config import PersonalConfig
from agentic_memory.autobiography_context import personal_context
from agentic_memory.autobiography_server import register_personal_tools
from agentic_memory.config import Config,load_config
from agentic_memory import server


class PersonalToolTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.personal=PersonalConfig(enabled=True,owner='sam',agent='assistant')
        self.cfg=Config(data_dir=self.root,personal=self.personal)
        self.principal=self.personal.principal(role='author',session_id='root-session')

    def tools(self,principal=None):
        service=MCPServer('test')
        get_store=register_personal_tools(service,self.cfg,principal=principal or self.principal)
        self.addCleanup(lambda:get_store().close())
        return {t.name:t for t in service._tool_manager.list_tools()}

    def test_disabled_mode_changes_no_tool_contract_and_creates_no_database(self):
        baseline=[(t.name,t.parameters) for t in server.mcp._tool_manager.list_tools()]
        cfg=replace(self.cfg,personal=PersonalConfig())
        self.assertIsNone(register_personal_tools(server.mcp,cfg))
        self.assertEqual([(t.name,t.parameters) for t in server.mcp._tool_manager.list_tools()],baseline)
        self.assertFalse(self.cfg.db_path.exists())
        self.assertFalse(any(name.startswith('autobiography') for name,_ in baseline))

    def test_author_can_edit_without_approval_and_reader_has_no_update_tool(self):
        tools=self.tools()
        self.assertEqual(len(tools),7)
        for tool in tools.values():
            self.assertFalse({'owner','agent','audience','role'} & set(tool.parameters.get('properties',{})))
        result=tools['autobiography_update'].fn(expected_revision=0,segment='Soul',title='Soul',
            body='I reconsider principles deliberately.',reason='Self-authored decision',idempotency_key='my-edit')
        self.assertTrue(result['ok'])
        record_id=result['result']['edits'][0]['record_id']
        self.assertEqual(tools['autobiography_get'].fn(record_id=record_id)['result']['revision'],1)
        readonly=self.tools(self.personal.principal())
        self.assertNotIn('autobiography_update',readonly)
        self.assertNotIn('autobiography_update',self.tools(self.personal.principal(role='dream',session_id='dream-session')))

    def test_lifecycle_tools_preserve_history_and_require_authority(self):
        tools = self.tools()
        result = tools['autobiography_update'].fn(expected_revision=0, segment='Identity', title='Identity',
            body='Original content', reason='Authored', idempotency_key='create')
        rid = result['result']['edits'][0]['record_id']
        result = tools['autobiography_transition'].fn(record_id=rid, expected_revision=1,
            state='outdated', reason='Contradicted', idempotency_key='invalidate')
        self.assertTrue(result['ok'])
        self.assertFalse(tools['autobiography_get'].fn(record_id=rid)['ok'])
        self.assertEqual(tools['autobiography_get'].fn()['result']['records'], [])
        self.assertEqual(tools['autobiography_get'].fn(record_id=rid, revision=1)['result']['body'], 'Original content')
        self.assertEqual(tools['autobiography_get'].fn(record_id=rid, include_inactive=True)['result']['state'], 'outdated')
        for role in ('reader', 'dream', 'reconciler'):
            readonly = self.tools(self.personal.principal(role=role, session_id='service'))
            self.assertNotIn('autobiography_transition', readonly)
        self.assertFalse(tools['autobiography_get'].fn(revision=1)['ok'])

    def test_context_is_explicit_stable_and_person_selection_is_bounded(self):
        store=AutobiographyStore(self.cfg.db_path,self.principal);self.addCleanup(store.close)
        self.assertEqual(personal_context(store)['status'],'uninitialized')
        for segment in ['Identity','Soul','User','Core','People','Relationships','Experiences','Reflections']:
            store.update(expected_revision=0,segment=segment,subject='core' if segment in ('Identity','Soul','User','Core') else 'sam',
                title=segment,body='A compact authored statement.',reason='Self-authorship',idempotency_key=segment)
        first=personal_context(store)
        self.assertEqual(first['status'],'ready')
        self.assertEqual([r['segment'] for r in first['manifest']['records']],['Soul','User','Identity','Core'])
        self.assertEqual(first['manifest']['generation'],personal_context(store)['manifest']['generation'])
        selected=personal_context(store,subjects=['sam'])
        self.assertEqual(len(selected['manifest']['records']),6)
        self.assertNotIn('Experiences:',selected['text'])
        self.assertEqual(selected['manifest']['token_accounting']['used'],len(selected['text'].encode()))

    def test_mandatory_core_never_disappears_silently_to_fit_budget(self):
        tools=self.tools()
        tools['autobiography_update'].fn(expected_revision=0,segment='Soul',title='Core',body='Important principle. '*100,
            reason='Self-authorship',idempotency_key='long')
        result=tools['autobiography_context'].fn(token_budget=256)
        self.assertFalse(result['ok']);self.assertEqual(result['error']['code'],'context_overflow')

    def test_configuration_requires_explicit_owner_and_trusted_author_session(self):
        with self.assertRaises(PersonalError):PersonalConfig.parse({'enabled':True})
        with self.assertRaises(PersonalError):self.personal.principal(role='author')
        with self.assertRaises(ValueError):PersonalConfig.parse({'enabled':'false'})
        with self.assertRaises(ValueError):PersonalConfig.parse({'context_budget':1})
        cfg_path=self.root/'settings.toml'
        cfg_path.write_text('[personal]\nenabled=true\nowner="sam"\nagent="assistant"\n')
        self.assertEqual(load_config(cfg_path).personal.principal().role,'reader')
        with self.assertRaises(PersonalError):register_personal_tools(MCPServer('x'),self.cfg,principal=replace(self.principal,owner='other'))

    def test_inbox_read_reports_uninitialized_extraction_without_provider_calls(self):
        result=self.tools()['autobiography_inbox'].fn()
        self.assertTrue(result['ok'])
        self.assertEqual(result['result']['candidates'],[])
