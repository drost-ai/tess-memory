"""Guided setup: a small web app on this machine that writes the settings.

``tess-memory setup`` starts it on 127.0.0.1 and opens the browser. It finds
the agents already on the machine, connects a model and an embedding model,
registers the memory server with those agents and runs a first ingestion.
What it writes are the same files the command line reads: a settings file and
a provider file. Nothing here is needed afterwards, and both files can still
be edited by hand.
"""
