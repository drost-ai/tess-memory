"""Role-bound personal MCP surface. Scope never comes from model arguments."""
from functools import wraps
from .autobiography import AutobiographyStore, PersonalError
from .autobiography_context import personal_context


def register_personal_tools(server, config, *, principal=None):
    if not config.personal.enabled:
        return None
    bound=principal or config.personal.principal()
    if (bound.owner,bound.agent)!=(config.personal.owner,config.personal.agent):
        raise PersonalError('forbidden','configured personal scope unavailable')
    if bound.audience!=config.personal.audience and not config.personal.multi_audience:
        raise PersonalError('forbidden','configured personal scope unavailable')
    opened=[]
    operation_services=[]
    def store():
        if not opened:
            opened.append(AutobiographyStore(config.db_path,bound,context_budget=config.personal.context_budget))
        return opened[0]
    def operations():
        if not operation_services:
            import atexit
            from .personal_operations import PersonalOperations
            operation_services.append(PersonalOperations(config,store()))
            atexit.register(operation_services[0].close)
        return operation_services[0]
    def expose(fn):
        @wraps(fn)
        def operation(*args,**kwargs):
            try:return {'ok':True,'result':fn(*args,**kwargs)}
            except PersonalError as exc:return {'ok':False,'error':{'code':exc.code,'message':str(exc)}}
        hidden_model=bound.role in ('follow_up_reconciler','topic_reconciler','care','research') and fn.__name__.startswith('autobiography_model_')
        hidden_candidates=bound.role in ('care','research') and fn.__name__=='autobiography_register_candidates'
        hidden_status=bound.role in ('follow_up_reconciler','topic_reconciler','care','research') and fn.__name__=='autobiography_operation_status'
        if not hidden_model and not hidden_candidates and not hidden_status:
            server.add_tool(operation,name=fn.__name__)
        return operation

    @expose
    def autobiography_get(record_id: str | None=None, segment: str | None=None, limit: int=20, after: str='', revision: int | None=None, include_inactive: bool=False) -> dict:
        """Read current authored memory in this configured scope; exact reads do not use a model."""
        if revision is not None and record_id is None:
            raise PersonalError('invalid', 'an explicit revision requires record_id')
        return store().get(record_id,revision,include_inactive=include_inactive) if record_id else store().list_records(segment=segment,limit=limit,after=after,include_inactive=include_inactive)

    @expose
    def autobiography_context(token_budget: int | None=None, subjects: list[str] | None=None) -> dict:
        """Get an explicitly bounded core and the exact loaded revision manifest. Overflow is visible."""
        budget=config.personal.context_budget if token_budget is None else token_budget
        if budget>config.personal.context_budget:
            raise PersonalError('budget','requested context exceeds the configured ceiling')
        return personal_context(store(),budget=budget,subjects=config.personal.context_subjects if subjects is None else subjects)

    @expose
    def autobiography_search(query: str, segment: str | None=None, limit: int=12) -> dict:
        """Search only this audience's authored autobiography; ordinary coding search remains separate."""
        return store().search(query,segment=segment,limit=limit)

    @expose
    def autobiography_history(record_id: str, before_revision: int | None=None, limit: int=20) -> dict:
        """Read complete prior authored revisions without replacing the current head."""
        return store().history(record_id,before=before_revision,limit=limit)

    @expose
    def autobiography_inbox(state: str='pending', limit: int=20, after: str='') -> dict:
        """Inspect evidence-linked proposals; a candidate is not an authored belief."""
        from .autobiography_inbox import PersonalInbox
        return PersonalInbox(store()).list(state=state,limit=limit,after=after)

    if bound.role in ('extractor','operator'):
        def workflow():
            from .extraction_workflow import ExtractionWorkflow
            return ExtractionWorkflow(config,store(),operations())

        @expose
        def autobiography_extraction_claim(operation_id: str, operation_fence: int, limit: int=8) -> dict:
            return workflow().claim(operation_id,operation_fence,limit)

        @expose
        def autobiography_extraction_read(operation_id: str, operation_fence: int, range_id: str, lease_token: str, offset: int=0, limit: int=12000) -> dict:
            return workflow().read(operation_id,operation_fence,range_id,lease_token,offset,limit)

        @expose
        def autobiography_extraction_candidate(operation_id: str, operation_fence: int, range_id: str, lease_token: str, candidate_key: str, request_id: str, expected_revision: int, action: str='submit', candidate: dict | None=None, candidate_kind: str='memory') -> dict:
            return workflow().candidate(operation_id,operation_fence,range_id,lease_token,candidate_key,request_id,expected_revision,action,candidate,candidate_kind)

        @expose
        def autobiography_extraction_finalize(operation_id: str, operation_fence: int, range_id: str, lease_token: str, request_id: str) -> dict:
            return workflow().finalize(operation_id,operation_fence,range_id,lease_token,request_id)

        @expose
        def autobiography_extraction_get(part_id: str) -> dict:
            """Service operation: recover a publication receipt after a lost acknowledgment."""
            from .autobiography_inbox import PersonalInbox
            return PersonalInbox(store()).part_status(part_id)

        @expose
        def autobiography_extraction_prepare(limit: int | None=None) -> dict:
            """Service operation: prepare changed source parts without editing authored state."""
            from .autobiography_extract import PersonalExtraction
            return PersonalExtraction(config,store()).prepare(limit=limit)

        @expose
        def autobiography_extraction_publish(part_id: str, candidates: list[dict], operation_id: str | None=None, operation_fence: int | None=None) -> dict:
            """Service operation: atomically publish source-grounded candidates and extraction progress."""
            from .autobiography_extract import PersonalExtraction
            admission=(lambda:operations()._operation(operation_id,operation_fence)) if bound.role=='extractor' else None
            if admission:admission()
            return PersonalExtraction(config,store()).publish(part_id,candidates,admission=admission)

        @expose
        def autobiography_extraction_fail(part_id: str, code: str) -> dict:
            """Service operation: retain failed work with bounded retry and dead-letter state."""
            from .autobiography_inbox import PersonalInbox
            PersonalInbox(store()).fail(part_id,code)
            return {'retained':True}

    if bound.role in ('extractor','dream','reconciler','follow_up_reconciler','topic_reconciler','care','research','operator'):
        @expose
        def autobiography_operation_claim(kind: str, manual: bool=False, request_deadline: float | None=None) -> dict:
            """Service operation: claim a due personal job without overlapping another worker."""
            return operations().claim(kind,manual=manual,request_deadline=request_deadline)

        @expose
        def autobiography_operation_get(operation_id: str) -> dict:
            """Service operation: recover a durable job outcome after a lost acknowledgment."""
            return operations().get_operation(operation_id)

        @expose
        def autobiography_operation_finish(operation_id: str, fence: int, outcome: str='completed', result: dict | None=None) -> dict:
            """Service operation: persist outcome and the next scheduled time."""
            return operations().finish(operation_id,fence,outcome=outcome,result=result)

        @expose
        def autobiography_operation_status() -> dict:
            """Service operation: read the schedule and conservative usage reservations."""
            return operations().status()

        @expose
        def autobiography_model_reserve(operation_id: str, fence: int, work_id: str, identity: str, provider: str, model: str,
                                        system: str, prompt: str, input_units: int, slots: int=1) -> dict:
            """Service operation: enforce daily/job/provider admission or reuse an exact response checkpoint."""
            return operations().reserve(operation_id,fence,work_id=work_id,identity=identity,provider=provider,model=model,system=system,prompt=prompt,input_units=input_units,slots=slots)

        @expose
        def autobiography_model_complete(operation_id: str, fence: int, request_id: str, response: dict | None=None, usage: dict | None=None, error: str | None=None) -> dict:
            """Service operation: retain the paid result before subsequent publication or cleanup."""
            return operations().complete_model(operation_id,fence,request_id,response=response,usage=usage,error=error)

        @expose
        def autobiography_model_reject(operation_id: str, fence: int, cache_key: str, code: str) -> dict:
            """Service operation: invalidate a structurally rejected proposal without replaying it forever."""
            return operations().reject_cache(operation_id,fence,cache_key,code)

    if bound.role in ('dream','reconciler','operator'):
        def reconcile_operation(operation_id,operation_fence):
            op=operations()._operation(operation_id,operation_fence)
            if op['kind'] not in ('dream','reconciler'):
                raise PersonalError('forbidden','reconciler operation required')
            return op

        @expose
        def autobiography_reconcile_claim(operation_id: str, operation_fence: int, initialize: bool=False) -> dict:
            from .dream_cycle import DreamCycle
            op=reconcile_operation(operation_id,operation_fence)
            return DreamCycle(config,store()).prepare(deadline=op['deadline'],initialize=initialize,
                admission=lambda:reconcile_operation(operation_id,operation_fence),full_lease=True)

        @expose
        def autobiography_reconcile_commit(operation_id: str, operation_fence: int, cycle_id: str, fence: int, lease_token: str, request_id: str, edits: list[dict], outcomes: list[dict]) -> dict:
            from .dream_cycle import DreamCycle
            return DreamCycle(config,store()).commit(cycle_id,fence,lease_token,idempotency_key=request_id,edits=edits,outcomes=outcomes,
                admission=lambda:reconcile_operation(operation_id,operation_fence))

        @expose
        def autobiography_reconcile_defer(operation_id: str, operation_fence: int, cycle_id: str, fence: int, lease_token: str, request_id: str, candidate_ids: list[str], reason: str, review_after: float | None=None) -> dict:
            from .dream_cycle import DreamCycle
            outcomes=[{'candidate_id':cid,'action':'deferred','reason':reason,**({'review_after':review_after} if review_after is not None else {})} for cid in candidate_ids]
            return DreamCycle(config,store()).commit(cycle_id,fence,lease_token,idempotency_key=request_id,edits=[],outcomes=outcomes,
                admission=lambda:reconcile_operation(operation_id,operation_fence))

        @expose
        def autobiography_cycle_prepare(operation_id: str | None=None, operation_fence: int | None=None, initialize: bool=False) -> dict:
            """Service operation: claim a bounded reflection batch with a fenced lease."""
            from .dream_cycle import DreamCycle
            deadline=operations()._operation(operation_id,operation_fence)['deadline'] if operation_id else None
            return DreamCycle(config,store()).prepare(deadline=deadline,initialize=initialize)

        @expose
        def autobiography_cycle_renew(cycle_id: str, fence: int, lease_token: str) -> dict:
            """Service operation: renew a live claim without extending the cycle deadline."""
            from .dream_cycle import DreamCycle
            return DreamCycle(config,store()).renew(cycle_id,fence,lease_token)

        @expose
        def autobiography_cycle_commit(cycle_id: str, fence: int, lease_token: str, idempotency_key: str, edits: list[dict], outcomes: list[dict]) -> dict:
            """Service operation: atomically validate and commit autonomous revisions and candidate dispositions."""
            from .dream_cycle import DreamCycle
            return DreamCycle(config,store()).commit(cycle_id,fence,lease_token,idempotency_key=idempotency_key,edits=edits,outcomes=outcomes)

        @expose
        def autobiography_cycle_finish(cycle_id: str, fence: int, lease_token: str, status: str='completed', error: str | None=None) -> dict:
            """Service operation: report committed work and release any unfinished candidates."""
            from .dream_cycle import DreamCycle
            return DreamCycle(config,store()).finish(cycle_id,fence,lease_token,status=status,error=error)

        @expose
        def autobiography_cycle_status(limit: int=10) -> dict:
            """Service operation: inspect actual cycle outcomes and commit receipts."""
            from .dream_cycle import DreamCycle
            return {'cycles':DreamCycle(config,store()).status(limit)}

    def activation():
        from .prospective_activation import ProspectiveActivation
        return ProspectiveActivation(config,store())

    def registers():
        if not config.personal.registers_enabled or not activation().status().get('active'):
            raise PersonalError('disabled','personal registers require enablement and prospective activation')
        from .personal_registers import PersonalRegisters
        return PersonalRegisters(store(),operations())

    if bound.role in ('extractor','follow_up_reconciler','topic_reconciler','care','research','operator'):
        @expose
        def autobiography_register_status() -> dict:
            result=activation().status()
            return {**result,'enabled':config.personal.registers_enabled,'activated':bool(result.get('active')),'policy_version':result.get('activation_version',0)}

    if bound.role == 'operator':
        @expose
        def autobiography_register_recover(invalidate_all: bool=False) -> dict:
            from .personal_recovery import recover_registers
            return recover_registers(store(),invalidate_all=invalidate_all)

        @expose
        def autobiography_register_activate(request_id: str, expected_policy_version: int=0) -> dict:
            return activation().activate(request_id,expected_policy_version)

        @expose
        def autobiography_register_suppress(kind: str, item_id: str, expected_revision: int, request_id: str, reason: str) -> dict:
            return registers().suppress(kind,item_id,expected_revision,request_id,reason)

    if bound.role in ('follow_up_reconciler','topic_reconciler','care','research','operator'):
        @expose
        def autobiography_register_candidates(kind: str, state: str='pending', limit: int=20, after: str='') -> dict:
            return registers().candidates(kind,state,limit,after)

        @expose
        def autobiography_register_list(kind: str, state: str='active', due_before: float | None=None, limit: int=20, after: str='') -> dict:
            return registers().list(kind,state,due_before,limit,after)

    if bound.role in ('follow_up_reconciler','topic_reconciler','operator'):
        @expose
        def autobiography_register_reconcile_claim(operation_id: str, operation_fence: int, kind: str, limit: int=20) -> dict:
            return registers().reconcile_claim(operation_id,operation_fence,kind,limit)

        @expose
        def autobiography_register_reconcile_commit(operation_id: str, operation_fence: int, kind: str, fence: int, lease_token: str, request_id: str, edits: list[dict], outcomes: list[dict]) -> dict:
            return registers().reconcile_commit(operation_id,operation_fence,kind,fence,lease_token,request_id,edits,outcomes)

    if bound.role in ('care','research','operator'):
        @expose
        def autobiography_register_claim(operation_id: str, operation_fence: int, kind: str, item_id: str, expected_revision: int, request_id: str) -> dict:
            return registers().claim(operation_id,operation_fence,kind,item_id,expected_revision,request_id)

        @expose
        def autobiography_register_validate_claim(operation_id: str, operation_fence: int, kind: str, item_id: str, fence: int, lease_token: str, expected_revision: int) -> dict:
            return registers().validate_claim(operation_id,operation_fence,kind,item_id,fence,lease_token,expected_revision)

        @expose
        def autobiography_register_progress(operation_id: str, operation_fence: int, kind: str, item_id: str, fence: int, lease_token: str, expected_revision: int, request_id: str, summary: str, findings: list | None=None, next_step: str='', review_after: float | None=None) -> dict:
            return registers().progress(operation_id,operation_fence,kind,item_id,fence,lease_token,expected_revision,request_id,summary,findings or [],next_step,review_after)

        @expose
        def autobiography_register_complete(operation_id: str, operation_fence: int, kind: str, item_id: str, fence: int, lease_token: str, expected_revision: int, request_id: str, outcome: str, summary: str, review_after: float | None=None, delivery_receipt: dict | None=None) -> dict:
            """Store an outcome; a supplied receipt is NOT proof of platform delivery."""
            return registers().complete(operation_id,operation_fence,kind,item_id,fence,lease_token,expected_revision,request_id,outcome,summary,review_after,delivery_receipt)

    if bound.role in ('author','operator'):
        @expose
        def autobiography_transition(record_id: str, expected_revision: int, state: str, reason: str, idempotency_key: str,
                                     superseded_by: str | None=None, superseded_by_revision: int | None=None) -> dict:
            """Append active/retired/outdated/superseded/retracted lifecycle history without rewriting evidence.

            Supersession requires an active replacement record and exact revision. Historical content remains accessible.
            """
            return store().transition(record_id,expected_revision=expected_revision,state=state,reason=reason,
                idempotency_key=idempotency_key,superseded_by=superseded_by,superseded_by_revision=superseded_by_revision)

        @expose
        def autobiography_update(expected_revision: int, title: str, body: str, reason: str, idempotency_key: str,
                                 record_id: str | None=None, segment: str | None=None, subject: str | None=None,
                                 attribution: str='agent_reflection', evidence: list[dict] | None=None, state: str='active') -> dict:
            """Autonomously author any segment, including Principles and Relationships. No user approval is needed.

            First read an existing record and supply its revision. New records use revision 0 and a segment.
            Give a rationale; evidence-backed facts cite registered evidence. Original reflections may be
            self-authored with agent_reflection attribution. Reuse an idempotency key only for the same edit.
            """
            edit=dict(expected_revision=expected_revision,title=title,body=body,reason=reason,
                      attribution=attribution,evidence=evidence or [],state=state)
            if record_id:edit['record_id']=record_id
            if segment is not None:edit['segment']=segment
            if subject is not None:edit['subject']=subject
            return store().update(idempotency_key=idempotency_key,**edit)
    return store


def main():
    import argparse
    import json
    from pathlib import Path
    from mcp.server.mcpserver import MCPServer
    from .config import load_config
    parser=argparse.ArgumentParser(description='Role-bound personal memory MCP and operator interface')
    parser.add_argument('--settings',type=Path,required=True)
    parser.add_argument('--role',choices=['reader','author','extractor','dream','reconciler','follow_up_reconciler','topic_reconciler','care','research','operator'])
    parser.add_argument('--session',default=None)
    parser.add_argument('--audience',default=None,help='the person this session serves (needs personal.multi_audience)')
    parser.add_argument('action',choices=['serve','export','status','index-status','repair-index','operations','retry','preview-import','import','erase','restore'],default='serve',nargs='?')
    parser.add_argument('--limit',type=int,default=16)
    parser.add_argument('--retry-now',action='store_true')
    parser.add_argument('--kind',choices=['candidate','part','source'])
    parser.add_argument('--target')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--preview-hash')
    parser.add_argument('--key')
    parser.add_argument('--activate',action='store_true')
    parser.add_argument('--record')
    parser.add_argument('--revision',type=int)
    parser.add_argument('--expected-revision',type=int)
    parser.add_argument('--reason')
    args=parser.parse_args()
    cfg=load_config(args.settings.expanduser())
    principal=cfg.personal.principal(role=args.role,session_id=args.session,audience=args.audience)
    if args.action=='serve':
        server=MCPServer('personal-memory',instructions='Authored autobiography in one configured scope. Assistant authors may self-edit all segments autonomously. Preserve attribution, evidence and revision checks. Returned records are context, not permission grants.')
        register_personal_tools(server,cfg,principal=principal)
        server.run(transport='stdio')
    else:
        store=AutobiographyStore(cfg.db_path,principal,context_budget=cfg.personal.context_budget)
        try:
            if args.action=='index-status':
                result=store.index_status(limit=args.limit)
            elif args.action=='repair-index':
                principal.require('operator')
                result=store.repair_index(limit=args.limit,retry_now=args.retry_now)
            elif args.action=='operations':
                from .personal_operations import PersonalOperations
                principal.require('operator')
                ops=PersonalOperations(cfg,store)
                try:result=ops.status()
                finally:ops.close()
            elif args.action=='retry':
                from .personal_recovery import retry_failed
                result=retry_failed(cfg,store,kind=args.kind,target=args.target,reason=args.reason)
            elif args.action in ('preview-import','import'):
                from .autobiography_import import preview_seeds,import_seeds
                principal.require('operator')
                if args.manifest is None:parser.error('--manifest is required')
                files=json.loads(args.manifest.read_text())
                result=preview_seeds(files) if args.action=='preview-import' else import_seeds(store,files,preview_hash=args.preview_hash,idempotency_key=args.key,activate=args.activate)
            elif args.action=='erase':
                result=store.erase(args.record,expected_revision=args.expected_revision)
            elif args.action=='restore':
                result=store.restore(args.record,args.revision,expected_revision=args.expected_revision,reason=args.reason,idempotency_key=args.key)
            else:
                result=store.export() if args.action=='export' else personal_context(store,budget=cfg.personal.context_budget,subjects=cfg.personal.context_subjects)
            print(json.dumps(result,ensure_ascii=False))
        finally:store.close()


if __name__=='__main__':
    main()
