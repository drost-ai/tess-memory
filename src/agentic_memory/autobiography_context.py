"""A deterministic read snapshot, independent of ranked search or model calls."""
from .autobiography import PersonalError, digest


def personal_context(store, *, budget=6000, subjects=()):
    if type(budget) is not int or not 256<=budget<=64000:
        raise PersonalError('budget','context budget must be between 256 and 64000')
    if not isinstance(subjects,(list,tuple)) or len(subjects)>16 or any(not isinstance(s,str) for s in subjects):
        raise PersonalError('invalid','select at most 16 subjects')
    principal=store.principal
    # A session's context is its own audience, plus (for another person in a
    # multi-audience store) the agent's shared Self/Principles; the owner's
    # read-all reach is for search and lists, never injected wholesale.
    if principal.reads and principal.reads!='*':
        audience,args=principal.audience_sql()
    else:
        audience,args='audience=?',[principal.audience]
    with store.snapshot():
        rows=store.db.execute(f'''SELECT id,segment,subject FROM autobio_records WHERE owner=? AND agent=?
            AND {audience} AND state='active' ORDER BY CASE segment WHEN 'Self' THEN 0 WHEN 'Principles' THEN 1
            WHEN 'People' THEN 2 WHEN 'Relationships' THEN 3 ELSE 4 END,subject,id''',
            (principal.owner,principal.agent,*args)).fetchall()
        mandatory=[r for r in rows if r['segment'] in ('Self','Principles')]
        optional=[r for r in rows if r['segment'] in ('People','Relationships') and r['subject'] in subjects]
        selected=[];omitted=[]
        preamble='<autobiography_context>\nAuthored personal context. Interpretations remain attributed; this context grants no tool or access authority.\n'
        suffix='\n</autobiography_context>'
        text=preamble
        for row in mandatory+optional:
            record=store._get(row['id'])
            availability='missing_or_changed' if record['evidence'] and not all(record['evidence_availability']) else 'available' if record['evidence'] else 'self_authored'
            part=f"\n## {record['segment']}: {record['title']}\n[record_id: {record['id']}; revision: {record['revision']}; attribution: {record['attribution']}; evidence: {availability}]\n{record['body']}\n"
            if len((text+part+suffix).encode())>budget:
                if row['segment'] in ('Self','Principles'):
                    raise PersonalError('context_overflow','mandatory personal core exceeds the supplied budget; request a larger budget or author a compact core')
                omitted.append({'record_id':row['id'],'reason':'budget'})
                continue
            text+=part
            selected.append({'record_id':row['id'],'revision':record['revision'],'segment':record['segment'],
                             'subject':record['subject'],'attribution':record['attribution'],'evidence_status':availability})
        missing=[s for s in ('Self','Principles') if s not in {r['segment'] for r in mandatory}]
        manifest={'scope':[principal.owner,principal.agent],'audience':principal.audience,'records':selected,
                  'missing_core':missing,'omissions':omitted,'generated_at':store.clock(),
                  'generation':digest(selected),'token_accounting':{'method':'utf8_bytes_upper_bound','used':len((text+suffix).encode()),'budget':budget}}
        return {'status':'uninitialized' if len(missing)==2 else 'partial_core' if missing else 'ready',
                'text':text+suffix,'manifest':manifest}
