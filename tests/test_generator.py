"""Generator tests: how server notes show up in the prompt and in the sources line."""

import os
import sys
import types

# generator.py builds its clients on import. A dummy key and a stand-in retriever
# keep these tests offline.
os.environ.setdefault("OPENAI_API_KEY", "test")
sys.modules.setdefault("retriever", types.SimpleNamespace(retrieve=None))

import generator

WIKI = {"title": "Rex", "url": "https://ark.wiki.gg/wiki/Rex", "text": "Rex text", "source": "wiki"}
NOTE = {"title": "Rates", "url": "", "text": "Taming speed: 5x", "source": "server"}


def test_wiki_only_prompt_is_unchanged():
    assert generator.system_prompt([WIKI]) == generator.SYSTEM_PROMPT
    assert generator.build_prompt("q", [WIKI]) == "Context:\n[1] Rex: Rex text\n\nQuestion: q"


def test_server_notes_are_labelled_and_add_the_server_rule():
    assert "[1] Server notes (Rates): Taming speed: 5x" in generator.build_prompt("q", [NOTE, WIKI])
    assert generator.system_prompt([NOTE, WIKI]) == generator.SYSTEM_PROMPT + generator.SERVER_RULE


def test_sources_line_links_wiki_pages_and_labels_notes():
    line = generator.source_links([NOTE, WIKI, NOTE])
    assert line == "Sources: Server notes (Rates) · [Rex](<https://ark.wiki.gg/wiki/Rex>)"


def test_split_sources_maps_numbers_back_to_chunks():
    text, used = generator.split_sources("Taming is 5x here.\nSources: 1", [NOTE, WIKI])
    assert text == "Taming is 5x here."
    assert used == [NOTE]
