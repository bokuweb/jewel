"""Pure component selection rules for Jewel export profiles."""

from __future__ import annotations

NER_COMPONENT = "ner"
PARSER_COMPONENT = "parser"
SENTENCIZER_COMPONENT = "sentencizer"
TOK2VEC_COMPONENT = "tok2vec"


def select_ner_components(
    pipe_names: list[str],
    *,
    uses_tok2vec_listener: bool,
    sentencizer_names: tuple[str, ...] = (),
) -> frozenset[str]:
    """Select the components required by Jewel's extraction-only runtime."""
    available = set(pipe_names)
    if NER_COMPONENT not in available:
        raise ValueError("NER profile requires a missing 'ner' component")
    if len(sentencizer_names) > 1:
        raise ValueError("NER profile supports at most one sentencizer component")
    if any(name not in available for name in sentencizer_names):
        raise ValueError("NER profile received an unknown sentencizer component")

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
    elif sentencizer_names:
        selected.add(sentencizer_names[0])
    elif SENTENCIZER_COMPONENT in available:
        selected.add(SENTENCIZER_COMPONENT)
    return frozenset(selected)
