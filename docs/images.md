# Image records

Tess Memory can remember the images people send to an agent. It never sees an image. The
agent's host keeps the files, has a model describe each one, and records the description
here, so images are found by the same search, and with the same embedding model, as
everything else the agent remembers.

An image record is like a [plan item](plan-board.md): it is written, not derived from a
transcript. Nothing in Tess Memory can rebuild one, and ingesting a session again never
touches it.

## What a record holds

| Field | |
|---|---|
| `ref` | The host's id for the image. Recording the same `ref` again replaces the record. |
| `description` | What a model wrote about the image. Required. |
| `path` | Where the host keeps the image file. An agent with a file-reading tool can open it. |
| `name` | The file name the image was sent with. |
| `session_id` | The session it was sent in, when known. |
| `project` | Where it is filed. When the session is already in the store, the session's project is used. |
| `sent_at`, `taken_at` | When it was sent, and when the camera says it was taken. |
| `facts` | Anything else the host read out of the image: an object, kept as given. |

## Recording from a host

A host records and removes images by running the command line with the store's settings,
one JSON object a line on standard input:

```bash
tess-memory images apply < operations.jsonl
```

```json
{"op": "record", "ref": "5f2c9a", "description": "A night photograph on a rooftop terrace...", "path": "/home/you/.agent/images/5f2c9a.webp", "name": "IMG_0042.jpg", "session_id": "0b9e...", "sent_at": "2026-11-20T23:10:00+00:00"}
{"op": "remove", "ref": "77aa01"}
```

It prints one JSON result a line, in order: `{"ref": ..., "ok": true, "embedded": true}`,
or `"ok": false` with the reason. One refused line does not stop the rest. `embedded` is
false when the description could not be embedded (the embedding model is being changed, or
its endpoint did not answer); the record is kept and found by its words until it is
recorded again or `tess-memory reembed` has run.

From Python, the same operations are `tess_memory.images.record_image`, `remove_image`
and `get_image`.

## Finding images

A host that shows its own image browser asks which images match, best first:

```bash
tess-memory images search "a woman in a red dress on a rooftop" --limit 60
```

It prints one `{"ref": ..., "score": ...}` a line. This is one lookup by words and meaning,
with no model call but the embedding of the query.


Each record is in the search index at level `image`. See
[Searching memory](searching.md#finding-images). `get_session` lists the images sent in a
session, so a memory of a conversation leads to its pictures.

Image records are private to their owner by default, like sessions. See
[Teams](teams.md).
