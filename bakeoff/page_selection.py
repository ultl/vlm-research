"""Shared page-selection helpers for bakeoff entrypoints."""
from __future__ import annotations


def _parse_pages_arg(values):
    if not values:
        return None
    pages = []
    for value in values:
        pages.extend(name.strip() for name in value.split(",") if name.strip())
    if not pages:
        raise ValueError("--pages must include at least one page name")
    return pages


def resolve_page_names(config_pages, pages_arg):
    """Return selected page names, validating CLI overrides against config."""
    selected = _parse_pages_arg(pages_arg)
    if selected is None:
        return list(config_pages)

    known = set(config_pages)
    unknown = [name for name in selected if name not in known]
    if unknown:
        raise ValueError(
            f"unknown page(s): {', '.join(unknown)}; "
            f"known pages: {', '.join(config_pages)}"
        )
    return selected
