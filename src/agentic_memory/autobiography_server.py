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
        server.add_tool(operation,name=fn.__name__)
        return operation

    @expose
    def autobiography_get(record_id: str | None=None, segment: str | None=None, limit: int=20, after: str='') -> dict:
        """Read current authored memory in this configured scope; exact reads do not use a model."""
        return store().get(record_id) if record_id else store().list_records(segment=segment,limit=limit,after=after)

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
        @expose
        def autobiography_extraction_get(part_id: str) -> dict:
            """Service operation: recover a publication receipt after a lost acknowledgment."""
            from .autobiography_inbox import PersonalInbox
            return PersonalInbox(store()).part_status(part_id)

        @expose
        def autobiography_extraction_prepare(limit: int=8) -> dict:
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

    if bound.role in ('extractor','dream','operator'):
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

    if bound.role in ('dream','operator'):
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

    if bound.role in ('author','operator'):
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
    parser.add_argument('--role',choices=['reader','author','extractor','dream','operator'])
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
