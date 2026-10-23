"""Synthetic labeled retrieval corpus for choosing the local embedding default.

Everything here is invented: no person, path or error comes from a real
installation. Each query names the one passage that answers it; every
category also carries near-miss distractors (the same entity with another
attribute, the same attribute for another entity) so a model has to match
meaning, not vocabulary. The long-context category buries the answer in the
middle of a long passage, which shows what a model's token limit truncates.

    python corpus.py --summary
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass

SEED = 20261022


@dataclass(frozen=True)
class Passage:
    id: str
    category: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    category: str
    text: str
    relevant: str


PEOPLE = ["Mara Okafor", "Theo Lindqvist", "Priya Raman", "Jonas Weber", "Aiko Tanaka", "Lucía Ferrer",
          "Sam Doyle", "Nadia Haddad", "Owen Price", "Ines Duarte", "Kofi Mensah", "Elena Petrova"]
FACTS = [  # (attribute, value phrase, question)
    ("tea", "drinks genmaicha every morning", "What does {first} drink in the morning?"),
    ("allergy", "is allergic to hazelnuts", "Which food is {first} allergic to?"),
    ("birthday", "was born on the 14th of March", "When is {first}'s birthday?"),
    ("instrument", "plays the cello in a community orchestra", "Which instrument does {first} play?"),
    ("commute", "cycles to work along the canal", "How does {first} get to work?"),
    ("pet", "has a grey cat called Pixel", "What is the name of {first}'s cat?"),
]
RELATIONS = [  # (relation, question)
    ("younger brother", "Who is {a}'s younger brother?"),
    ("business partner", "Who does {a} run the business with?"),
    ("mentor", "Who mentored {a}?"),
    ("neighbour", "Who lives next door to {a}?"),
]
CITIES = ["Lisbon", "Osaka", "Tallinn", "Montreal", "Accra", "Valparaíso", "Leeds", "Brno"]
TECH = [
    ("The gateway listens on port 8420 and binds only to 127.0.0.1 unless a secure remote policy is configured.",
     "Which port does the gateway listen on by default?"),
    ("Compaction starts when a session's input passes the profile's compactionThresholdTokens, here 300000 tokens.",
     "At what token count does compaction begin?"),
    ("Scheduled jobs store a rule in the owner's IANA zone and resolve the next fire time in UTC across daylight saving.",
     "How are scheduled job times kept correct across daylight saving changes?"),
    ("The task daemon is started by the engine runtime and is replaced once by the watchdog if it dies.",
     "What happens when the task daemon crashes?"),
    ("Remembered browsers are forgotten after 30 days without use and must re-authorize 180 days after the terminal proof.",
     "How long does a remembered browser stay signed in?"),
    ("Partial output state is finalized when the delivery receipt hash matches the streamed text hash.",
     "When is partial output state marked committed?"),
    ("The provider registry reads only providerProfiles; credentials come from an auth envVar or a managed credentialRef.",
     "Where do provider profiles get their credentials?"),
    ("Session ingestion runs hourly and is skipped as busy while another coordinator holds the supervisor lock.",
     "Why would an ingestion run report busy?"),
]
TECH_DISTRACTORS = [
    "The web app's development server listens on port 5173 and proxies /api to the gateway.",
    "Compaction output is capped by compactionOutputTokens, 30000 for this profile.",
    "Scheduled reminders created by the agent are refused when their time is already in the past.",
    "The browser stream reconnects with exponential backoff capped at eight seconds.",
    "A management session expires after 30 minutes idle or 8 hours in total.",
    "Assistant text that never streamed leaves no hash in the partial output state.",
    "Registry records must not embed credentials or Authorization headers.",
    "Ingestion cleanup waits 90 seconds for a cancelled worker before forcing it to stop.",
]
ERRORS = [
    ("ProviderWireError: OpenAI request failed with status 400 — model not supported with a ChatGPT account",
     "openai 400 model not supported chatgpt account error"),
    ("KeyError: 'time' raised by describe() for an hourly schedule rule", "keyerror time when describing hourly schedule"),
    ("PermissionError: test isolation denied: socket operation (including Unix)", "test isolation denied socket permission error"),
    ("sqlite3.OperationalError: database is locked while writing the inbound ledger", "database locked error writing ledger"),
    ("AttributeError: 'NoneType' object has no attribute 'start' in setup lifecycle", "nonetype has no attribute start lifecycle crash"),
    ("ValueError: Provider profile changes require explicit session configuration", "provider profile change requires explicit session configuration"),
]
ERROR_DISTRACTORS = [
    "ProviderWireError: OpenAI request failed with status 429 — rate limited",
    "KeyError: 'minute' raised while normalizing an interval rule",
    "PermissionError: test isolation denied: outside-temp mutation",
    "sqlite3.OperationalError: no such table: inbound",
    "AttributeError: 'NoneType' object has no attribute 'stop' during shutdown",
    "ValueError: Unknown provider profile requested by the session",
]
PATHS = [
    ("Owner configuration lives in settings/config/config.yaml under the runtime home.", "where is the config.yaml file kept"),
    ("Provider profiles are defined in settings/providers.yaml.", "which file defines provider profiles"),
    ("Remembered browser hashes are stored in settings/management-state.json.", "where are remembered browser hashes stored"),
    ("Each engine session's transcript is kept in sessions/<id>/full-transcript-with-tools.jsonl.", "where is a session's full transcript"),
    ("The packaged system prompt seed is host_app/_assets/system_prompt.md.", "path of the packaged system prompt seed"),
]
PATH_DISTRACTORS = [
    "settings/config/config.yaml.example shows every engine knob with its default.",
    "providers.yaml.bak is left behind by older migration scripts.",
    "settings/.management-server.json holds the live server's rendezvous epoch.",
    "sessions/<id>.messages.jsonl holds the compact message log.",
    "settings/system_prompt.md is the runtime copy the owner may customize.",
]
MULTILINGUAL = [  # (passage, query, language)
    ("Mara prefiere las reuniones por la mañana, nunca después de las cuatro de la tarde.", "When does Mara prefer meetings?", "es"),
    ("Theo hat im Mai seinen Job bei der Bahn gekündigt und arbeitet jetzt als Tischler.", "What does Theo do for work now?", "de"),
    ("アイコは毎週土曜日に陶芸教室に通っています。", "What does Aiko do on Saturdays?", "ja"),
    ("Priya a déménagé à Montréal pour être plus proche de sa sœur.", "Why did Priya move to Montreal?", "fr"),
    ("Jonas está a aprender português para a viagem ao Brasil em dezembro.", "Why is Jonas learning Portuguese?", "pt"),
    ("Нина каждое утро бегает пять километров вдоль реки.", "How far does Nina run each morning?", "ru"),
]
MULTILINGUAL_DISTRACTORS = [
    "Mara prefiere el té verde al café.",
    "Theo hat im Juni Urlaub in Norwegen gemacht.",
    "アイコは日曜日に図書館で働いています。",
    "Priya a une sœur qui vit à Toronto.",
    "Jonas viaja ao Brasil todos os anos em julho.",
    "Нина каждый вечер читает перед сном.",
]
BULK = [  # everyday notes about the same people: plausible, never an answer
    "{person} mentioned a new {thing} at dinner.",
    "{person} asked about the {topic} appointment next week.",
    "{person}'s colleague recommended a book about {topic}.",
    "{person} said the {thing} they saw in {city} was overpriced.",
    "{person} is planning a trip to {city} in {month}.",
    "Remind {person} about the {topic} paperwork before {month}.",
    "{person} and a friend argued about the best {thing} for a small flat.",
    "{person} skipped the {topic} meeting because of a cold.",
]
BULK_TECH = [
    "The {component} logs to {path} and retries {count} times before reporting a failure.",
    "A {component} restart drops in-flight {topic} work, which the next run picks up again.",
    "Set {component} verbosity in {path}; the default keeps {count} days of logs.",
]
THINGS = ["espresso machine", "bicycle", "laptop", "guitar", "balcony garden", "camera", "kettle", "rain jacket"]
TOPICS = ["tax", "dentist", "release", "school", "insurance", "gym", "visa", "conference"]
MONTHS = ["January", "March", "May", "July", "September", "November"]
COMPONENTS = ["channel supervisor", "memory server", "scheduler", "web bundle builder", "doctor command", "telegram adapter"]
LOG_PATHS = ["logs/gateway.log", "logs/memory.log", "run/memory/server.log", "logs/channels.log"]
FILLER = ("The weekly sync covered release logistics, a long discussion of test containment and the state of the "
          "documentation backlog. Several people raised the cost of flaky integration runs and agreed to revisit the "
          "schedule after the next milestone. ")


def build() -> tuple[list[Passage], list[Query]]:
    rng = random.Random(SEED)
    passages: list[Passage] = []
    queries: list[Query] = []

    def add(category: str, text: str) -> str:
        pid = f"p{len(passages):04d}"
        passages.append(Passage(pid, category, text))
        return pid

    def ask(category: str, text: str, relevant: str) -> None:
        queries.append(Query(f"q{len(queries):04d}", category, text, relevant))

    for person in PEOPLE:
        first = person.split()[0]
        for attribute, value, question in rng.sample(FACTS, 3):
            ask("personal", question.format(first=first), add("personal", f"{person} {value}."))
    for index, person in enumerate(PEOPLE):
        relation, question = RELATIONS[index % len(RELATIONS)]
        other = PEOPLE[(index + 5) % len(PEOPLE)]
        city = CITIES[index % len(CITIES)]
        pid = add("relationship", f"{other} is {person.split()[0]}'s {relation}; they met in {city}.")
        ask("relationship", question.format(a=person.split()[0]), pid)
        add("relationship", f"{person.split()[0]} visited {city} with a friend from work last spring.")
    for text, question in TECH:
        ask("technical", question, add("technical", text))
    for text in TECH_DISTRACTORS:
        add("technical", text)
    for text, question in ERRORS:
        ask("error", question, add("error", text))
    for text in ERROR_DISTRACTORS:
        add("error", text)
    for text, question in PATHS:
        ask("path", question, add("path", text))
    for text in PATH_DISTRACTORS:
        add("path", text)
    for text, question, _language in MULTILINGUAL:
        ask("multilingual", question, add("multilingual", text))
    for text in MULTILINGUAL_DISTRACTORS:
        add("multilingual", text)
    for index, (text, question) in enumerate(TECH[:4] + PATHS[:2]):
        # The answer sits after ~400 words of filler: models that truncate
        # at 256 or 512 tokens may never see it.
        before = FILLER * (8 + index)
        ask("long_context", question, add("long_context", before + text + " " + FILLER * 3))
    for _ in range(360):
        add("distractor", rng.choice(BULK).format(person=rng.choice(PEOPLE), thing=rng.choice(THINGS), topic=rng.choice(TOPICS),
                                                   city=rng.choice(CITIES), month=rng.choice(MONTHS)))
    for _ in range(120):
        add("distractor", rng.choice(BULK_TECH).format(component=rng.choice(COMPONENTS), path=rng.choice(LOG_PATHS),
                                                        count=rng.choice([3, 5, 7, 14]), topic=rng.choice(TOPICS)))
    return passages, queries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--json", action="store_true")
    options = parser.parse_args()
    passages, queries = build()
    if options.json:
        print(json.dumps({"passages": [asdict(p) for p in passages], "queries": [asdict(q) for q in queries]}, ensure_ascii=False))
        return
    counts: dict[str, list[int]] = {}
    for passage in passages:
        counts.setdefault(passage.category, [0, 0])[0] += 1
    for query in queries:
        counts.setdefault(query.category, [0, 0])[1] += 1
    for category, (p, q) in counts.items():
        print(f"{category:14s} passages={p:3d} queries={q:3d}")
    print(f"{'total':14s} passages={len(passages):3d} queries={len(queries):3d}")


if __name__ == "__main__":
    main()
