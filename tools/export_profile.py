"""Pure component selection rules for Jewel export profiles."""

from __future__ import annotations

NER_COMPONENT = "ner"
PARSER_COMPONENT = "parser"
TOK2VEC_COMPONENT = "tok2vec"


def select_ner_components(
    pipe_names: list[str],
    *,
    uses_tok2vec_listener: bool,
) -> frozenset[str]:
    """Select the components required by Jewel's extraction-only runtime."""
    available = set(pipe_names)
    if NER_COMPONENT not in available:
        raise ValueError("NER profile requires a missing 'ner' component")

    needs_upstream = PARSER_COMPONENT in available or uses_tok2vec_listener
    if needs_upstream and TOK2VEC_COMPONENT not in available:
        raise ValueError(
            "NER profile requires an upstream tok2vec component for "
            "the parser or NER listener"
        )

    selected = {NER_COMPONENT}
    if needs_upstream:
        selected.add(TOK2VEC_COMPONENT)
    if PARSER_COMPONENT in available:
        selected.add(PARSER_COMPONENT)
    return frozenset(selected)
