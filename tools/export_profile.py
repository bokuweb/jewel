"""Pure component selection rules for Jewel export profiles."""

from __future__ import annotations

NER_COMPONENT = "ner"
PARSER_COMPONENT = "parser"
SENTENCIZER_COMPONENT = "sentencizer"
SENTER_COMPONENT = "senter"
TOK2VEC_COMPONENT = "tok2vec"


def select_ner_components(
    pipe_names: list[str],
    *,
    uses_tok2vec_listener: bool,
    sentencizer_names: tuple[str, ...] = (),
    senter_names: tuple[str, ...] = (),
) -> frozenset[str]:
    """Select the components required by Jewel's extraction-only runtime."""
    available = set(pipe_names)
    if NER_COMPONENT not in available:
        raise ValueError("NER profile requires a missing 'ner' component")
    sentence_components = sentencizer_names + senter_names
    if len(sentence_components) > 1:
        raise ValueError(
            "NER profile supports at most one sentence boundary component"
        )
    if any(name not in available for name in sentence_components):
        raise ValueError("NER profile received an unknown sentence boundary component")

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
    elif sentence_components:
        selected.add(sentence_components[0])
    elif SENTENCIZER_COMPONENT in available:
        selected.add(SENTENCIZER_COMPONENT)
    elif SENTER_COMPONENT in available:
        selected.add(SENTER_COMPONENT)
    return frozenset(selected)
