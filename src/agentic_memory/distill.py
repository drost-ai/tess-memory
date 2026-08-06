"""LLM distillation: transcripts → structured session digests, profiles, digests.

All prompts demand JSON-only output; parsing is defensive (fence stripping,
balanced-brace extraction, one retry with an explicit correction request).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import AgentCoreBackend

MEMORY_KINDS = ("decision", "fact", "preference", "solution", "problem", "task_state")

SESSION_SYSTEM = """\
You distill transcripts of AI coding-agent sessions into structured memory for
future retrieval. You write for a future agent that has NO other context: every
memory must be self-contained, name concrete files/projects/technologies, and
never use pronouns whose referent lives outside the memory itself. Output only
valid JSON — no prose, no markdown fences."""

SESSION_PROMPT = """\
Below is a transcript of a coding-agent session (project: {project}, date: {date}).

Produce a JSON object with exactly these fields:
{{
  "title": "short imperative title of what the session was about",
  "summary": "3-6 sentences: what was attempted, how, and what state it ended in",
  "outcome": "one sentence: completed / partially done / abandoned / blocked, and why",
  "memories": [
    {{
      "content": "one self-contained, durable fact worth remembering across sessions",
      "kind": "decision|fact|preference|solution|problem|task_state",
      "importance": 1-10,
      "tags": ["lowercase", "keywords"]
    }}
  ]
}}

Memory guidance:
- 3 to 12 memories. Prefer fewer, denser memories over many shallow ones.
- kind=decision: an architectural/technical choice and its rationale.
- kind=solution: a hard-won fix (error message -> root cause -> fix). These are gold.
- kind=preference: how the user likes things done (style, workflow, tone).
- kind=problem: a known open issue or pitfall in the project.
- kind=task_state: where the work was left, and the concrete next step.
- importance rubric: 9-10 = changes how future work must be done; 6-8 = saves
  significant future effort; 3-5 = useful context; 1-2 = trivia (omit these).
- Skip routine tool churn, greetings, and anything derivable from the code itself.

TRANSCRIPT:
{transcript}
"""

PROFILE_SYSTEM = """\
You maintain rolling project profiles for an AI coding agent's long-term memory.
A profile re-grounds a future agent that has never seen the project. Output only
valid JSON — no prose, no markdown fences."""

PROFILE_PROMPT = """\
Project: {project}
Working directory: {cwd}

Session history, newest first (title | date | summary | outcome):
{sessions}

Produce a JSON object:
{{
  "profile": "A markdown profile of this project, max ~350 words: what it is, its architecture/stack, key decisions made across sessions, current state of the work, known problems, and the user's established preferences for this project."
}}
"""

DIGEST_SYSTEM = """\
You write weekly activity digests for an AI coding agent's long-term memory.
Output only valid JSON — no prose, no markdown fences."""

DIGEST_PROMPT = """\
ISO week starting {week_start}. Sessions that week (project | title | summary | outcome):
{sessions}

Produce a JSON object:
{{
  "digest": "One dense paragraph (max ~150 words): what was worked on this week across projects, what was accomplished, what was left open."
}}
"""


SEGMENT_PROMPT = """\
Below is segment {idx} of {total} from one long coding-agent session
(project: {project}, date: {date}).{coverage_note} Segments overlap slightly,
so content near the edges may also appear in the neighboring segment.

Produce a JSON object:
{{
  "summary": "4-8 sentences: what happened in THIS segment (work attempted, findings, state changes)",
  "memories": [
    {{"content": "...", "kind": "decision|fact|preference|solution|problem|task_state",
      "importance": 1-10, "tags": ["..."]}}
  ]
}}

Memory guidance (same rubric as full-session distillation):
- 2 to 8 memories for this segment; self-contained; name concrete files/projects/tech.
- kinds: decision (choice + rationale), solution (error -> root cause -> fix),
  preference (how the user likes things done), problem (open issue/pitfall),
  task_state (where work stands), fact (durable context).
- importance: 9-10 changes future work; 6-8 saves real effort; 3-5 context; 1-2 omit.

SEGMENT:
{segment}
"""

REDUCE_PROMPT = """\
A long coding-agent session (project: {project}, date: {date}) was distilled in
{total} overlapping segments. Below are the per-segment summaries (in order) and
all candidate memories extracted from them. Because segments overlap, some
candidates describe the same thing twice.

Produce the final JSON object for the WHOLE session:
{{
  "title": "short imperative title of what the session was about",
  "summary": "4-8 sentences covering the arc of the whole session: what was attempted, how it evolved, and what state it ended in",
  "outcome": "one sentence: completed / partially done / abandoned / blocked, and why",
  "memories": [
    {{"content": "...", "kind": "decision|fact|preference|solution|problem|task_state",
      "importance": 1-10, "tags": ["..."]}}
  ]
}}

Rules for memories: consolidate duplicates and near-duplicates from overlapping
segments into single, best-phrased entries; keep at most 15, preferring the
highest-importance and most durable ones; keep only one task_state (the final
one — earlier task states are superseded).

SEGMENT SUMMARIES:
{summaries}

CANDIDATE MEMORIES (JSON):
{candidates}
"""


@dataclass
class DistilledMemory:
    content: str
    kind: str
    importance: float  # 0..1
    tags: list[str] = field(default_factory=list)


@dataclass
class SessionDigest:
    title: str
    summary: str
    outcome: str
    memories: list[DistilledMemory]


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError(f"no JSON object in LLM output: {text[:200]!r}")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced JSON in LLM output")


def _call_json(llm: AgentCoreBackend, system: str, prompt: str) -> dict:
    raw = llm.complete(system=system, prompt=prompt)
    try:
        return _extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        retry = (
            prompt
            + "\n\nYour previous reply was not valid JSON. Reply again with ONLY the JSON object."
        )
        raw = llm.complete(system=system, prompt=retry)
        return _extract_json(raw)


def _parse_memories(data: dict, min_importance: float = 0.25) -> list[DistilledMemory]:
    memories = []
    for m in data.get("memories", []):
        content = (m.get("content") or "").strip()
        if not content:
            continue
        kind = m.get("kind") if m.get("kind") in MEMORY_KINDS else "fact"
        try:
            imp = float(m.get("importance", 5)) / 10.0
        except (TypeError, ValueError):
            imp = 0.5
        imp = min(1.0, max(0.1, imp))
        if imp < min_importance:  # drop trivia (default: rated 1-2 of 10)
            continue
        tags = [str(t).lower() for t in (m.get("tags") or []) if isinstance(t, (str, int))][:8]
        memories.append(DistilledMemory(content=content, kind=kind, importance=imp, tags=tags))
    return memories


def _digest_from(data: dict, min_importance: float = 0.25) -> SessionDigest:
    return SessionDigest(
        title=(data.get("title") or "").strip() or "Untitled session",
        summary=(data.get("summary") or "").strip(),
        outcome=(data.get("outcome") or "").strip(),
        memories=_parse_memories(data, min_importance),
    )


def distill_session(
    llm: AgentCoreBackend, project: str, date: str, transcript_text: str,
    min_importance: float = 0.25,
) -> SessionDigest:
    data = _call_json(
        llm,
        SESSION_SYSTEM,
        SESSION_PROMPT.format(project=project, date=date, transcript=transcript_text),
    )
    return _digest_from(data, min_importance)


def segment_text(
    text: str, segment_chars: int, overlap: int, max_segments: int,
    hard_cap: int = 600_000,
) -> tuple[list[str], bool]:
    """Split text into overlapping sliding windows.

    Grows the window when the default size would exceed max_segments; if even
    hard_cap-sized windows cannot cover the text within max_segments, the
    windows are spaced evenly across the text (gaps elided) and the second
    return value is True to flag partial coverage.
    """
    if len(text) <= segment_chars:
        return [text], False
    step = segment_chars - overlap
    n = -(-(len(text) - overlap) // step)
    if n > max_segments:
        segment_chars = min(hard_cap, -(-len(text) // max_segments) + overlap)
        step = segment_chars - overlap
        n = -(-(len(text) - overlap) // step)
    if n <= max_segments:
        return [text[i * step : i * step + segment_chars] for i in range(n)], False
    # Even sampling: max_segments windows spread across the whole text.
    span = len(text) - segment_chars
    starts = [round(i * span / (max_segments - 1)) for i in range(max_segments)]
    return [text[s : s + segment_chars] for s in starts], True


def distill_session_segmented(
    llm: AgentCoreBackend,
    project: str,
    date: str,
    transcript_text: str,
    segment_chars: int,
    overlap: int,
    max_segments: int,
    hard_cap: int = 600_000,
    min_importance: float = 0.25,
) -> SessionDigest:
    """Map-reduce distillation for sessions too large for one pass."""
    segments, partial = segment_text(
        transcript_text, segment_chars, overlap, max_segments, hard_cap
    )
    coverage_note = (
        " NOTE: the session was too large to cover fully; segments are evenly"
        " sampled windows, so there are gaps between them."
        if partial
        else ""
    )
    summaries: list[str] = []
    candidates: list[dict] = []
    for i, seg in enumerate(segments, 1):
        data = _call_json(
            llm,
            SESSION_SYSTEM,
            SEGMENT_PROMPT.format(
                idx=i, total=len(segments), project=project, date=date,
                coverage_note=coverage_note, segment=seg,
            ),
        )
        summaries.append(f"[segment {i}/{len(segments)}] {(data.get('summary') or '').strip()}")
        for m in _parse_memories(data, min_importance):
            candidates.append(
                {"content": m.content, "kind": m.kind,
                 "importance": round(m.importance * 10), "tags": m.tags}
            )
    data = _call_json(
        llm,
        SESSION_SYSTEM,
        REDUCE_PROMPT.format(
            project=project, date=date, total=len(segments),
            summaries="\n".join(summaries),
            candidates=json.dumps(candidates, ensure_ascii=False),
        ),
    )
    return _digest_from(data, min_importance)


def distill_profile(
    llm: AgentCoreBackend, project: str, cwd: str, session_lines: list[str]
) -> str:
    data = _call_json(
        llm,
        PROFILE_SYSTEM,
        PROFILE_PROMPT.format(project=project, cwd=cwd or "?", sessions="\n".join(session_lines)),
    )
    return (data.get("profile") or "").strip()


def distill_week(
    llm: AgentCoreBackend, week_start: str, session_lines: list[str]
) -> str:
    data = _call_json(
        llm,
        DIGEST_SYSTEM,
        DIGEST_PROMPT.format(week_start=week_start, sessions="\n".join(session_lines)),
    )
    return (data.get("digest") or "").strip()
