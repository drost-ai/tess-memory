"""Personal candidate read surface; publication is independent of coding ingestion."""
from .autobiography import PersonalError


class PersonalInbox:
    def __init__(self,store):
        self.store=store

    def list(self,*,state='pending',limit=20,after=''):
        if state not in ('pending','claimed','accepted','merged','dismissed','deferred','stale','failed'):
            raise PersonalError('invalid','unknown candidate state')
        with self.store.transaction(write=False):
            exists=self.store.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='autobio_candidates'").fetchone()
            if not exists:
                return {'candidates':[],'next_cursor':None,'extraction_initialized':False}
            p=self.store.principal
            rows=self.store.db.execute('''SELECT * FROM autobio_candidates WHERE owner=? AND agent=? AND audience=?
                AND state=? AND id>? ORDER BY id LIMIT ?''',(p.owner,p.agent,p.audience,state,after,max(1,min(int(limit),100))+1)).fetchall()
            limit=max(1,min(int(limit),100))
            return {'candidates':[dict(r) for r in rows[:limit]],'next_cursor':rows[limit-1]['id'] if len(rows)>limit else None,'extraction_initialized':True}
