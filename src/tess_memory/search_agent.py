"""Bounded agentic retrieval: a model forages the store, returns curated evidence.

search_memory runs this by default. The pool is seeded with a static hybrid
search of the caller's query; then a small model judges coverage, fans out
reformulated queries, follows session links surfaced in results, and finally
selects the evidence worth returning. The agent curates — it never answers.
Synthesis stays with the caller, who has the conversation context.

Every failure degrades to the static seed results, never to an error: memory
retrieval must not become less reliable by becoming smarter.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from . import retrieval
from .config import Config
from .distill import _extract_json
from .embed import Embedder
from .llm import EngineBackend
from .store import MemoryStore
from .freshness import session_freshness

AGENT_SYSTEM = """\
You are the retrieval agent inside a long-term memory system for AI coding
sessions. A caller (another AI agent, mid-conversation with the user) issued a
search request. Your job is to gather the best EVIDENCE from the memory store.
You do NOT answer the request — you curate raw memories for the caller to read.

The store holds, per past session: distilled memories (kinds: decision,
solution, preference, problem, task_state, fact), session summaries, raw
activity chunks (exact commands/errors/code), rolling project profiles, and
weekly digests. It also holds authored plan items (level 'plan': vision/
epics/slices/tasks/notes with stable ids and live status) — the user's
intended plan, written by agents, not derived from history.
It may also hold image records (level 'image'): images people sent, each as
a detailed description of what is visible, with the image's id, the path of
its file and the session it was sent in. Descriptions never name people or
places; who and where come from the session.

Each round, reply with ONLY a JSON object — one of:
  {"action": "search", "queries": ["...", "..."]}
      Run hybrid semantic+keyword searches. Vary the angle: exact error
      strings, file paths, project/technology names, synonyms, adjacent
      framings (a thing's origin may be recorded under a different project
      than its current home).
  {"action": "get_session", "session_ids": ["..."]}
      Pull full session digests (summary, outcome, all memories) for session
      ids already visible in the evidence pool.
  {"action": "finish", "keep": ["<ids>"], "note": "..."}
      Final selection: pool ids worth returning, most relevant first. The
      optional note (one sentence) flags gaps, e.g. "no memory of X found".

Judgment guidance:
- Judge coverage against the request, not against result count: does the
  evidence actually cover every part of what was asked? Follow leads —
  session ids, predecessor projects, systems named inside results.
- Prefer distilled memories and session summaries; reach for chunks when the
  request wants exact commands, flags, paths, or error text.
- Redundant evidence is cost, not safety: drop near-duplicates at finish.
- Stop when another round would not improve the evidence; 2-3 rounds is
  typical. Finish decisively.
- Copy session_ids exactly as quoted in evidence, including every prefix and
  suffix. Never infer an id from a title or kind. If an excerpt already answers
  the request, finish directly; expanding a session is optional.
- If none of the candidates supports the request, finish with keep: [].
  Similarity alone is not evidence. Prefer current decisions over explicitly
  superseded history, and respect the live status of authored plan items.
- A rejected proposal or an injected/fabricated measurement does not establish
  an accepted decision or a measured result. If those are the only candidates
  for a request seeking an actual result, keep: [] and explain the limitation
  in note. Such records can still be relevant when the request asks about the
  proposal or injection itself.
"""

ROUND_PROMPT = """\
SEARCH REQUEST (verbatim from the caller):
{query}

Constraints in force on every search: {constraints}
Round {rnd} of {max_rounds}.{must_finish}

EVIDENCE POOL ({n_pool} items; content truncated for display — the caller \
receives full text of whatever you keep):
{pool}

SEARCHES ALREADY RUN: {queries}
SESSIONS ALREADY EXPANDED: {sessions}
"""


@dataclass
class SearchTrace:
    rounds: int = 0
    queries: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)
    note: str = ""
    fallback_used: bool = False
    model_finished: bool = False
    failures: int = 0
    invalid_actions: int = 0


def _session_evidence(store: MemoryStore, session_id: str, cfg=None) -> dict | None:
    s = store.get_session(session_id)
    if not s:
        return None
    mems = store.memories_for_session(session_id)
    mem_lines = "\n".join(f"  - [{m['kind']}] {m['content']}" for m in mems)
    content = (
        f"{s['title']}\nSummary: {s['summary']}\nOutcome: {s['outcome']}"
        + (f"\nMemories:\n{mem_lines}" if mem_lines else "")
    )
    return {
        "id": f"session:{session_id}",
        "score": None,
        "level": "session",
        "kind": "full-session",
        "content": content,
        "importance": None,
        "ts": s["ended_at"],
        "project": s["project"],
        "session_id": session_id,
        "session_title": s["title"],
        "freshness": session_freshness(store, session_id, cfg),
    }


def _render_pool(pool: dict[str, dict], snippet: int = 450) -> str:
    if not pool:
        return "(empty)"
    lines = []
    for key, h in pool.items():
        where = h["project"] or "-"
        when = (h["ts"] or "")[:10]
        anchor = f" | session_id={json.dumps(h['session_id'])}" if h["session_id"] else ""
        text = " ".join(h["content"].split())
        if len(text) > snippet:
            text = text[:snippet] + "…"
        freshness = h.get('freshness', {}).get('status', 'unverified')
        lines.append(f"#{key} [{h['level']}/{h['kind']} | {where} | {when}{anchor} | freshness={freshness}]\n  {text}")
    return "\n".join(lines)


def _action_json(llm: EngineBackend, prompt: str) -> dict:
    raw = llm.complete(system=AGENT_SYSTEM, prompt=prompt)
    try:
        return _extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        raw = llm.complete(
            system=AGENT_SYSTEM,
            prompt=prompt
            + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON action object.",
        )
        return _extract_json(raw)


def agentic_search(
    cfg: Config,
    store: MemoryStore,
    embedder: Embedder,
    llm: EngineBackend,
    query: str,
    project: str | None = None,
    level: str | None = None,
    kind: str | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int | None = None,
) -> tuple[list[dict], SearchTrace]:
    """Run the bounded retrieval loop. Returns (evidence, trace).

    Raises only if even the static seed search fails; LLM trouble mid-loop
    falls back to auto-selecting the best of whatever the pool holds.
    """
    cap = limit if limit is not None else cfg.search_evidence_limit
    trace = SearchTrace()
    if cap <= 0 or not store.has_memories(
        project=project, levels=(level,) if level else None,
        kinds=(kind,) if kind else None, after=after, before=before,
    ):
        trace.note = "No memories in the requested scope."
        return [], trace

    def run_static(q: str) -> list[dict]:
        return retrieval.search(
            store, embedder, q,
            project=project,
            levels=(level,) if level else None,
            kinds=(kind,) if kind else None,
            after=after, before=before,
            limit=cfg.search_per_query_limit,
        )

    pool: dict[str, dict] = {}

    def add_hits(hits: list[dict]) -> None:
        for h in hits:
            pool.setdefault(str(h["id"]), h)

    add_hits(run_static(query))
    trace.queries.append(query)

    constraints = ", ".join(
        f"{k}={v}"
        for k, v in (
            ("project", project), ("level", level), ("kind", kind),
            ("after", after), ("before", before),
        )
        if v
    ) or "none"

    def auto_select() -> list[dict]:
        trace.fallback_used = True
        # Deliberately-expanded sessions outrank mid-tier hits; scored hits
        # keep their fused ranking.
        ranked = sorted(
            pool.values(),
            key=lambda h: -(h["score"] if h["score"] is not None else 0.75),
        )
        return ranked[:cap]

    feedback = ''
    for rnd in range(1, cfg.search_max_rounds + 1):
        trace.rounds = rnd
        last = rnd == cfg.search_max_rounds
        prompt = ROUND_PROMPT.format(
            query=query,
            constraints=constraints,
            rnd=rnd,
            max_rounds=cfg.search_max_rounds,
            must_finish=(
                ' This is the final round: you MUST reply with action "finish".'
                if last else ""
            ),
            n_pool=len(pool),
            pool=_render_pool(pool),
            queries="; ".join(trace.queries),
            sessions="; ".join(trace.sessions) or "(none)",
        )
        if feedback:
            prompt += '\nPREVIOUS ACTION FEEDBACK: ' + feedback
        try:
            action = _action_json(llm, prompt)
        except Exception as e:  # noqa: BLE001 — degrade, never fail retrieval
            trace.failures += 1
            trace.note = 'Retrieval model unavailable; static candidate selection used.'
            print(f"[search-agent] round {rnd} failed ({e}); auto-selecting",
                  file=sys.stderr)
            return auto_select(), trace

        act = action.get("action")
        if act == "search" and not last:
            queries = [str(q) for q in (action.get("queries") or []) if str(q).strip()]
            queries = queries[: cfg.search_queries_per_round]
            if not queries:
                return auto_select(), trace
            for q in queries:
                add_hits(run_static(q))
                trace.queries.append(q)
        elif act == "get_session" and not last:
            # A full session includes other levels/kinds/dates; it cannot
            # honor these narrower filters. Keep the filtered evidence pool.
            if level or kind or after or before:
                trace.invalid_actions += 1
                if trace.invalid_actions > 1:
                    return auto_select(), trace
                feedback = 'Whole-session expansion cannot honor these filters. Search within the current constraints or finish with evidence IDs already in the pool.'
                continue
            sids = action.get("session_ids") or []
            if isinstance(sids, str):
                sids = [sids]
            sids = [str(s) for s in sids][: cfg.search_queries_per_round]
            expanded = False
            visible = {h["session_id"] for h in pool.values() if h["session_id"]}
            for sid in sids:
                if sid in trace.sessions or sid not in visible:
                    continue
                ev = _session_evidence(store, sid, cfg)
                if ev and (not project or ev["project"] == project):
                    pool[ev["id"]] = ev
                    trace.sessions.append(sid)
                    expanded = True
            if not expanded:
                trace.invalid_actions += 1
                if trace.invalid_actions > 1:
                    return auto_select(), trace
                feedback = ('No requested session was expanded. Copy an exact, unexpanded session_id from the evidence, '
                            'or finish directly. Available session IDs: ' + json.dumps(sorted(visible)[:20]))
                continue
            feedback = ''
        elif act == "finish":
            trace.note = str(action.get("note") or "").strip()
            selection = action.get("keep")
            if not isinstance(selection, list):
                return auto_select(), trace
            trace.model_finished = True
            if not selection:
                return [], trace
            keep, seen = [], set()
            for k in selection:
                key = str(k).lstrip("#").strip()
                if key in pool and key not in seen:
                    keep.append(pool[key])
                    seen.add(key)
            return (keep[:cap] if keep else auto_select()), trace
        else:
            # Unknown action, or a non-finish action on the forced-final round.
            return auto_select(), trace

    return auto_select(), trace
