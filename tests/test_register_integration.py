"""Mixed extraction atomicity, role exposure and host-owned cadence (temp DBs)."""
from dataclasses import replace
import json
import unittest
from unittest.mock import patch
from test_extraction_workflow import WorkflowTests
from test_autobiography_extract import ExtractionTests
from tess_memory.autobiography import PersonalError
from tess_memory.personal_registers import PersonalRegisters
from tess_memory.autobiography_server import register_personal_tools


class RegisterIntegrationTests(unittest.TestCase):
    candidate=ExtractionTests.candidate
    setUp=WorkflowTests.setUp

    def enable(self):
        self.cfg=replace(self.cfg,personal=replace(self.cfg.personal,registers_enabled=True))
        self.flow.config=self.cfg
        self.ops.config=self.cfg
        class Policy:
            def validate_evidence(inner,source_id,generation,start,end,quote):
                return dict(source_id=source_id,generation=generation,start=start,end=start+len(quote),quote=quote,activation_version=1)
        self.flow._prospective=Policy()

    def proposal(self):
        return {'title':'Check preference','body':'Follow up on concise updates.','quote':'I prefer concise updates.',
                'key':'concise','attribution':'user_statement','due_at':2000000000}

    def test_disabled_and_immutable_kind(self):
        with self.assertRaises(PersonalError) as error:
            self.flow.candidate(*self.args,'follow','r1',0,candidate=self.proposal(),candidate_kind='follow_up')
        self.assertEqual(error.exception.code,'disabled')
        self.enable()
        self.flow.candidate(*self.args,'follow','r1',0,candidate=self.proposal(),candidate_kind='follow_up')
        with self.assertRaises(PersonalError):
            self.flow.candidate(*self.args,'follow','r2',1,action='update',candidate=self.candidate(),candidate_kind='memory')
        with self.assertRaises(PersonalError) as error:
            self.flow.candidate(*self.args,'follow','r1',0,candidate=self.proposal(),candidate_kind='topic')
        self.assertEqual(error.exception.code,'idempotency_mismatch')

    def test_mixed_finalize_rolls_back_and_replays(self):
        self.enable()
        self.flow.candidate(*self.args,'memory','m',0,candidate=self.candidate())
        self.flow.candidate(*self.args,'follow','f',0,candidate=self.proposal(),candidate_kind='follow_up')
        with patch.object(PersonalRegisters,'publish_candidate',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):self.flow.finalize(*self.args,'finish')
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_candidates').fetchone()[0],0)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_register_candidates').fetchone()[0],0)
        row=self.store.db.execute('SELECT state,receipt FROM autobio_extraction_parts WHERE id=?',(self.range['range_id'],)).fetchone()
        self.assertEqual(row['state'],'pending');self.assertIsNone(row['receipt'])
        receipt=self.flow.finalize(*self.args,'finish')
        self.assertEqual(len(receipt['register_candidate_ids']),1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_candidates').fetchone()[0],1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_register_candidates').fetchone()[0],1)
        self.assertEqual(self.flow.finalize(*self.args,'finish'),receipt)
        self.assertGreater(receipt['cursor']['end'],0)

    def test_register_operation_cooldown_does_not_skip_calendar_tick(self):
        self.enable()
        self.store.principal=replace(self.store.principal,role='operator')
        now=[2000000000.0];self.store.clock=lambda:now[0]
        with patch('tess_memory.prospective_activation.ProspectiveActivation.status',return_value={'active':True}):
            for kind,interval in [('care',3600),('research',21600)]:
                start=now[0]
                op=self.ops.claim(kind)
                self.assertEqual(op['status'],'running')
                now[0]=start+1800
                self.ops.finish(op['operation_id'],op['fence'])
                now[0]=start+interval
                self.assertEqual(self.ops.claim(kind)['status'],'running')

    def test_role_tools_are_distinct(self):
        class Server:
            def __init__(self):self.tools={}
            def add_tool(self,fn,name):self.tools[name]=fn
        for role in ('care','research','follow_up_reconciler','topic_reconciler'):
            server=Server()
            register_personal_tools(server,self.cfg,principal=replace(self.store.principal,role=role))
            self.assertNotIn('autobiography_reconcile_commit',server.tools)
            self.assertNotIn('autobiography_register_activate',server.tools)
            self.assertIn('autobiography_operation_claim',server.tools)
            self.assertNotIn('autobiography_operation_status',server.tools)
            self.store.principal=replace(self.store.principal,role=role)
            with self.assertRaises(PersonalError):self.ops.status()
            self.assertEqual('autobiography_register_claim' in server.tools,role in ('care','research'))
            self.assertEqual('autobiography_register_reconcile_claim' in server.tools,role.endswith('reconciler'))


class ProspectiveWorkflowTests(unittest.TestCase):
    def setUp(self):
        from test_personal_sources import PersonalSourceTests
        PersonalSourceTests.setUp(self)
        self.session=lambda *a,**kw:PersonalSourceTests.session(self,*a,**kw)
        self.append=lambda *a,**kw:PersonalSourceTests.append(self,*a,**kw)
        self.cfg=replace(self.cfg,personal=replace(self.cfg.personal,registers_enabled=True))
        self.store.principal=replace(self.store.principal,role='operator')
        self.now=1800000000.0;self.store.clock=lambda:self.now

    def test_each_bound_role_can_claim_and_finish_without_index_write(self):
        from tess_memory.prospective_activation import ProspectiveActivation
        class Server:
            def __init__(self):self.tools={}
            def add_tool(self,fn,name):self.tools[name]=fn
        self.session(messages=[('user','baseline')])
        ProspectiveActivation(self.cfg,self.store).activate('activate')
        for role in ('follow_up_reconciler','topic_reconciler','care','research'):
            server=Server()
            register_personal_tools(server,self.cfg,principal=replace(self.store.principal,role=role,session_id=role))
            with patch('tess_memory.autobiography.AutobiographyStore.repair_index',side_effect=AssertionError('new role index mutation')):
                response=server.tools['autobiography_operation_claim'](kind=role,manual=True,request_deadline=2000000000)
            self.assertTrue(response['ok'],response)
            self.assertEqual(response['result']['status'],'running')
            op=response['result']
            result=server.tools['autobiography_operation_finish'](op['operation_id'],op['fence'])
            self.assertTrue(result['ok'],result)

    def test_real_activation_pending_tail_then_mixed_publish(self):
        from tess_memory.prospective_activation import ProspectiveActivation
        from tess_memory.personal_operations import PersonalOperations
        from tess_memory.extraction_workflow import ExtractionWorkflow
        path,body=self.session(messages=[('user','old unprocessed tail')])
        activation=ProspectiveActivation(self.cfg,self.store)
        activation.activate('activate')
        self.now+=20
        self.append(body,[('assistant','old answer'),('user','Please check my appointment.'),('assistant','Noted')],stamp=__import__('datetime').datetime.fromtimestamp(self.now,__import__('datetime').timezone.utc).isoformat())
        ops=PersonalOperations(self.cfg,self.store);self.addCleanup(ops.close)
        flow=ExtractionWorkflow(self.cfg,self.store,ops)
        op=ops.claim('extractor');auth=(op['operation_id'],op['fence'])
        ranges=flow.claim(*auth)['ranges']
        eligible=[r for r in ranges if 'Please check' in r['text']][0]
        self.assertTrue(eligible['prospective']['eligible_spans'])
        args=(*auth,eligible['range_id'],eligible['lease_token'])
        proposal=dict(title='Appointment',body='Check appointment',quote='Please check my appointment.',attribution='user_statement',key='appointment')
        flow.candidate(*args,'follow','f',0,candidate=proposal,candidate_kind='follow_up')
        memory={k:v for k,v in proposal.items() if k!='key'};memory['segment']='People'
        flow.candidate(*args,'memory','m',0,candidate=memory)
        flow.candidate(*args,'topic','t',0,candidate={**proposal,'question':'How can appointment preparation improve?'},candidate_kind='topic')
        receipt=flow.finalize(*args,'final')
        self.assertEqual(len(receipt['register_candidate_ids']),2)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_candidates').fetchone()[0],1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM autobio_register_candidates').fetchone()[0],2)
        old=[r for r in ranges if 'old unprocessed tail' in r['text']][0]
        with self.assertRaises(PersonalError):
            flow.candidate(*auth,old['range_id'],old['lease_token'],'bad','bad',0,candidate={**proposal,'quote':'old unprocessed tail'},candidate_kind='follow_up')
        exported=self.store.export()
        self.assertEqual(len(exported['registers']['candidates']),2)
        evidence=json.loads(self.store.db.execute('SELECT evidence FROM autobio_candidates').fetchone()[0])
        authored=self.store.update(expected_revision=0,segment='People',title='Appointment',body='Check appointment',reason='Test source erasure',idempotency_key='author',evidence=evidence)
        record_id=authored['edits'][0]['record_id']
        erased=self.store.erase(record_id,expected_revision=1)
        self.assertEqual(erased['register_derivatives']['candidates'],2)
        self.assertEqual(self.store.export()['registers']['candidates'],[])
