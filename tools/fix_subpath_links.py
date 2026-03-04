#!/usr/bin/env python3
"""
Post-process built Hugo HTML to fix absolute links that don't include the
deployment subpath. This handles raw HTML inside markdown content that bypasses
Hugo's render hooks.

Usage: python3 tools/fix_subpath_links.py <public_dir> <base_path>
  public_dir : path to Hugo's built output directory (e.g. public/)
  base_path  : deployment subpath without trailing slash (e.g. /cpp)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def build_valid_top_level(public_dir: Path) -> set[str]:
    """Return the set of top-level names in public/ (files and dirs)."""
    return {item.name for item in public_dir.iterdir()}


def is_valid_deployment_url(url: str, base_path: str, top_level: set[str]) -> bool:
    """
    Return True if `url` is already a valid deployment URL:
    it starts with base_path/ and the next path component matches
    something that actually exists at the top level of public/.
    """
    prefix = base_path + "/"
    if not url.startswith(prefix):
        return False
    remainder = url[len(prefix):]          # e.g. "cpp/ranges/" or "css/main.css"
    first_component = remainder.split("/")[0]  # "cpp", "css", ""
    if not first_component:
        return True                        # URL == base_path/ (site home)
    return first_component in top_level


def fix_html(content: str, base_path: str, top_level: set[str]) -> str:
    """
    Rewrite href/src/data-* attribute values that are absolute paths but are
    not valid deployment URLs, by prepending base_path.
    Handles both quoted (href="/path") and unquoted (href=/path) attributes.
    """
    attrs_re = r"(?:href|src|data-card-href|data-pin-url|data-min-standard-path)"
    pattern = re.compile(
        rf"({attrs_re})"   # group 1: attribute name
        r"(=)"             # group 2: equals sign
        r'("?)'            # group 3: optional opening quote
        r"(/[^\">\s#]*)"   # group 4: absolute URL path (starts with /)
        r'("?)',           # group 5: optional closing quote
        re.ASCII,
    )

    def replacer(m: re.Match) -> str:
        attr, eq, q1, url, q2 = m.groups()
        if url.startswith("//"):              # protocol-relative – skip
            return m.group(0)
        if is_valid_deployment_url(url, base_path, top_level):
            return m.group(0)               # already correct
        return f"{attr}{eq}{q1}{base_path}{url}{q2}"

    return pattern.sub(replacer, content)


def process_directory(public_dir: Path, base_path: str) -> None:
    top_level = build_valid_top_level(public_dir)
    print(f"Deployment base : {base_path}/")
    print(f"Top-level names : {sorted(top_level)}")

    changed = 0
    for html_file in sorted(public_dir.rglob("*.html")):
        original = html_file.read_text(encoding="utf-8", errors="replace")
        fixed = fix_html(original, base_path, top_level)
        if fixed != original:
            html_file.write_text(fixed, encoding="utf-8")
            changed += 1
            print(f"  fixed: {html_file.relative_to(public_dir)}")
    print(f"Done – {changed} file(s) updated.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    public_dir = Path(sys.argv[1])
    base_path = sys.argv[2].rstrip("/")
    if not public_dir.is_dir():
        print(f"Error: {public_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    process_directory(public_dir, base_path)
