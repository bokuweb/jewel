"""Pure component selection rules for Jewel export profiles."""

from __future__ import annotations


def resolve_tok2vec_listener_upstream(
    upstream: str,
    tok2vec_names: tuple[str, ...],
) -> str:
    """Resolve spaCy's wildcard listener to one concrete tok2vec component."""
    if upstream != "*":
        return upstream
    if len(tok2vec_names) != 1:
        raise ValueError(
            "NER profile requires exactly one tok2vec component to resolve "
            "a wildcard listener"
        )
    return tok2vec_names[0]


def resolve_transformer_listener_upstream(
    upstream: str,
    transformer_names: tuple[str, ...],
) -> str:
    """Resolve spaCy's wildcard listener to one concrete transformer."""
    if upstream != "*":
        return upstream
    if len(transformer_names) != 1:
        raise ValueError(
            "NER profile requires exactly one transformer component to resolve "
            "a wildcard listener"
        )
    return transformer_names[0]


def select_ner_components(
    pipe_names: list[str],
    *,
    ner_names: tuple[str, ...] = ("ner",),
    parser_names: tuple[str, ...] = (),
    sentencizer_names: tuple[str, ...] = (),
    senter_names: tuple[str, ...] = (),
    entity_ruler_names: tuple[str, ...] = (),
    tok2vec_upstreams: dict[str, str] | None = None,
    transformer_upstreams: dict[str, str] | None = None,
) -> frozenset[str]:
    """Select the components required by Jewel's extraction-only runtime."""
    available = set(pipe_names)
    if len(ner_names) != 1:
        raise ValueError("NER profile requires exactly one NER component")
    if len(parser_names) > 1:
        raise ValueError("NER profile supports at most one parser component")
    sentence_components = sentencizer_names + senter_names
    if len(sentence_components) > 1:
        raise ValueError(
            "NER profile supports at most one sentence boundary component"
        )
    named_components = (
        ner_names + parser_names + sentence_components + entity_ruler_names
    )
    if any(name not in available for name in named_components):
        raise ValueError("NER profile received an unknown component")
    selected = {ner_names[0]}
    if parser_names:
        selected.add(parser_names[0])
    elif sentence_components:
        selected.add(sentence_components[0])
    selected.update(entity_ruler_names)

    for component_name, upstream_name in (tok2vec_upstreams or {}).items():
        if component_name not in selected:
            continue
        if upstream_name not in available:
            raise ValueError(
                f"NER profile requires missing upstream tok2vec "
                f"component {upstream_name!r}"
            )
        selected.add(upstream_name)
    for component_name, upstream_name in (transformer_upstreams or {}).items():
        if component_name not in selected:
            continue
        if upstream_name not in available:
            raise ValueError(
                f"NER profile requires missing upstream transformer "
                f"component {upstream_name!r}"
            )
        selected.add(upstream_name)
    return frozenset(selected)
