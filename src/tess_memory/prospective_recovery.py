"""Bounded, read-only recovery planning. There is deliberately NO apply API.

Published parts are terminal: eligibility fixes must not reset their states or
source cursors. A later explicitly authorized register-only recovery worker can
consume a reviewed manifest, with fresh generation/proof checks and its own
idempotency ledger. Never feed these ranges through memory extraction again.
"""
from copy import copy
from dataclasses import replace
import json

from .autobiography import PersonalError, digest
from .native_creation import native_creation_proof
from .prospective_activation import _stamp


def plan_native_recovery(policy, expected_receipts):
    """Review at most 11 explicitly selected register-empty publications; no DB writes.

    expected_receipts maps range IDs to digest(parsed original receipt). A caller
    must collect/review that allowlist separately, never sweep all publications.
    Register-empty publication is not proof of why the original model returned nothing;
    this plan reports the measurable old/new eligibility delta only.
    """
    if (not isinstance(expected_receipts, dict) or not 0 < len(expected_receipts) <= 11
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in expected_receipts.items())):
        raise PersonalError('invalid', 'explicit receipt allowlist must contain 1..11 ranges')
    legacy = copy(policy)
    legacy.config = replace(policy.config, personal=replace(policy.config.personal, native_creation_ledger=None))
    entries = []
    for range_id, receipt_hash in sorted(expected_receipts.items()):
        row = policy.store.db.execute('SELECT * FROM autobio_extraction_parts WHERE id=?', (range_id,)).fetchone()
        if row is None or row['state'] != 'published':
            raise PersonalError('conflict', 'recovery requires a published range')
        receipt = json.loads(row['receipt'] or 'null')
        result = json.loads(row['result'] or 'null')
        drafts = json.loads(row['drafts'])
        if (not isinstance(receipt, dict) or digest(receipt) != receipt_hash
            or receipt.get('range_id') != range_id or receipt.get('published') is not True
            or not isinstance(receipt.get('candidate_ids'), list) or receipt.get('register_candidate_ids', []) != []
            or not isinstance(result, list) or len(result) != len(receipt['candidate_ids'])
            or not isinstance(drafts, dict)
            or any(not isinstance(d, dict) or d.get('candidate_kind', 'memory') != 'memory' for d in drafts.values())):
            raise PersonalError('conflict', 'recovery requires an unchanged register-empty publication')
        if policy.store.db.execute("SELECT 1 FROM sqlite_master WHERE name='autobio_register_candidates'").fetchone():
            if policy.store.db.execute('''SELECT 1 FROM autobio_register_candidates
                WHERE owner=? AND agent=? AND audience=? AND range_id=?''', (*policy._scope, range_id)).fetchone():
                raise PersonalError('conflict', 'register candidates already exist for range')
        source, md, ds = policy._source(row['source_id'], row['generation'])
        args = (row['source_id'], row['generation'], row['start'], row['end'])
        if legacy.eligible_spans(*args):
            raise PersonalError('conflict', 'range was already eligible under legacy fences')
        spans = policy.eligible_spans(*args)
        if not spans:
            raise PersonalError('invalid_evidence', 'no newly proven native spans')
        saved = policy._baseline(policy.status()['activation_version'], md['path'])
        current = policy._capture(ds, md['path'], rewrite_fence=saved[0]['byte_highwater'])[0]
        proof = native_creation_proof(policy.config, ds, current, policy.status()['activated_at'], saved[1], _stamp)
        if proof is None:
            raise PersonalError('invalid_evidence', 'native authority changed during planning')
        entries.append({'range_id': range_id, 'source_id': source['id'], 'generation': row['generation'],
                        'start': row['start'], 'end': row['end'], 'receipt_hash': receipt_hash,
                        'preserved_memory_candidate_ids': receipt['candidate_ids'],
                        'publication_hash': digest([receipt, result, drafts]),
                        'range_text_hash': digest(row['text']), 'native_creation_proof_hash': digest(proof),
                        'baseline_hash': digest(saved), 'source_metadata_hash': digest(md),
                        'eligible_spans': spans})
    manifest = {'version': 1, 'mode': 'dry-run-register-only', 'scope': list(policy._scope),
                'activation': policy.status(), 'ranges': entries,
                'apply_supported': False, 'preserve_memory_publications': True}
    return {**manifest, 'manifest_hash': digest(manifest)}
