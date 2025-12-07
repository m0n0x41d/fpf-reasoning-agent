"""
FPF Specification Index - Parse and index FPF spec for section retrieval.

Pure functions for building and querying the FPF index.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache


@dataclass
class FPFSection:
    """A section in the FPF specification."""
    pattern_id: str
    title: str
    status: str
    start_line: int
    end_line: int
    keywords: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class FPFIndexStats:
    """Statistics about the parsed FPF index."""
    total_sections: int = 0
    parts_found: list[str] = field(default_factory=list)
    toc_entries: int = 0
    body_sections: int = 0
    missing_content: list[str] = field(default_factory=list)


# Multiple patterns to match section headers - ordered by specificity
SECTION_HEADER_PATTERNS = [
    # "## A.1 Holonic Foundation: Entity → Holon [A]" or "## B.3.4 — Evidence Decay [A]"
    re.compile(r'^##\s+(?:Pattern\s+)?([A-Z]\d?\.[\d.]+)\s*[—–:-]?\s*(.+?)\s*(?:\[.\])?\s*$'),
    # "### A.1.1 — `U.BoundedContext`: The Semantic Frame"
    re.compile(r'^###\s+(?:\*\*)?(?:Pattern\s+)?([A-Z]\d?\.[\d.]+)\s*[—–:-]?\s*(.+?)(?:\*\*)?\s*(?:\[.\])?\s*$'),
    # "## B.3.5 · **CT2R‑LOG — Working-Model Relations**"
    re.compile(r'^##\s+([A-Z]\d?\.[\d.]+)\s*[·]\s*\*\*(.+?)\*\*'),
    # "### 1) Intent" style inside sections - skip these
    # Table rows: "| A.1 | **Holonic Foundation...**"
    re.compile(r'^\|\s*\*?\*?([A-Z]\d?\.[\d.]+)\*?\*?\s*\|\s*\*\*(.+?)\*\*'),
    # Alternative: Pattern with status marker like "## A.1 Title \[A]"
    re.compile(r'^##\s+([A-Z]\d?\.[\d.]+)\s+(.+?)\s+\\\[([A-Z])\\\]'),
    # Catch-all for simpler formats
    re.compile(r'^#{2,3}\s+([A-Z]\d?\.[\d.]+)\s+(.+)$'),
]

# Pattern to extract Part headers
PART_PATTERN = re.compile(r'^#\s*\*?\*?Part\s+([A-Z])\*?\*?\s*[—–·:-]\s*(.+)$', re.IGNORECASE)

# Pattern ID normalizer - handles variations like "A1.1" vs "A.1.1"
PATTERN_ID_NORMALIZE = re.compile(r'^([A-Z])(\d)?(\.[\d.]+)$')


def normalize_pattern_id(pid: str) -> str:
    """
    Normalize pattern ID to canonical form.

    "A1" -> "A.1", "A1.1" -> "A.1.1", "A.1" stays "A.1"
    """
    pid = pid.strip().upper()

    # Already normalized
    if re.match(r'^[A-Z]\.\d+(?:\.\d+)*$', pid):
        return pid

    # Handle "A1" -> "A.1"
    if re.match(r'^[A-Z]\d+$', pid):
        return pid[0] + '.' + pid[1:]

    # Handle "A1.1" -> "A.1.1"
    match = re.match(r'^([A-Z])(\d)(\.[\d.]+)?$', pid)
    if match:
        part = match.group(1)
        first_digit = match.group(2)
        rest = match.group(3) or ''
        return f"{part}.{first_digit}{rest}"

    return pid


def parse_fpf_spec(spec_path: Path) -> dict[str, FPFSection]:
    """
    Parse FPF specification and build section index.

    Returns dict mapping pattern_id to FPFSection.
    """
    if not spec_path.exists():
        return {}

    lines = spec_path.read_text(encoding='utf-8', errors='replace').split('\n')

    stats = FPFIndexStats()
    sections: dict[str, FPFSection] = {}
    current_part = ""

    # First pass: extract TOC sections with metadata
    toc_sections = _extract_toc_sections(lines)
    stats.toc_entries = len(toc_sections)

    # Second pass: find actual section locations in body
    for i, line in enumerate(lines):
        # Track current part
        part_match = PART_PATTERN.match(line)
        if part_match:
            current_part = part_match.group(1)
            if current_part not in stats.parts_found:
                stats.parts_found.append(current_part)
            continue

        # Look for section headers
        for pattern in SECTION_HEADER_PATTERNS:
            match = pattern.match(line)
            if match:
                raw_pattern_id = match.group(1)
                pattern_id = normalize_pattern_id(raw_pattern_id)
                title = match.group(2).strip() if match.lastindex >= 2 else ""

                # Clean up title
                title = re.sub(r'\*\*', '', title)
                title = re.sub(r'\s*\[.\]\s*$', '', title)
                title = re.sub(r'^\s*`|`\s*$', '', title)
                title = title.strip(' —–:-')

                if not title:
                    continue

                if pattern_id not in sections:
                    sections[pattern_id] = FPFSection(
                        pattern_id=pattern_id,
                        title=title,
                        status="",
                        start_line=i,
                        end_line=i,  # Will be updated
                    )
                    stats.body_sections += 1
                else:
                    # Update start line if this is the actual section (not TOC)
                    # TOC is typically in first ~250 lines
                    if i > 250:
                        sections[pattern_id].start_line = i
                break

    # Third pass: determine end lines
    sorted_sections = sorted(sections.values(), key=lambda s: s.start_line)
    for i, section in enumerate(sorted_sections):
        if i + 1 < len(sorted_sections):
            section.end_line = sorted_sections[i + 1].start_line - 1
        else:
            section.end_line = len(lines) - 1

    # Merge TOC info
    for pid, toc_info in toc_sections.items():
        normalized_pid = normalize_pattern_id(pid)
        if normalized_pid in sections:
            sections[normalized_pid].keywords = toc_info.get('keywords', [])
            sections[normalized_pid].status = toc_info.get('status', '')
            sections[normalized_pid].dependencies = toc_info.get('dependencies', [])
        else:
            # Section in TOC but not found in body - track for debugging
            stats.missing_content.append(normalized_pid)

    stats.total_sections = len(sections)

    # Log statistics
    print(f"[FPF Index] Parsed {stats.total_sections} sections from body, "
          f"{stats.toc_entries} TOC entries, "
          f"Parts: {', '.join(sorted(stats.parts_found))}")

    if stats.missing_content and len(stats.missing_content) < 10:
        print(f"[FPF Index] Note: {len(stats.missing_content)} TOC entries without body content")

    return sections


def _extract_toc_sections(lines: list[str]) -> dict[str, dict]:
    """Extract section info from Table of Contents."""
    toc_sections = {}

    for line in lines[:600]:  # TOC is typically in first 600 lines
        # Match TOC rows like: | A.1 | **Title** | Stable | keywords... | deps... |
        # Also handle: | **A.1** | ... or | A.1.1 | ...
        match = re.match(
            r'^\|\s*\*?\*?([A-Z]\d?\.[\d.]+)\*?\*?\s*\|\s*\*\*(.+?)\*\*\s*\|\s*(\w+)\s*\|(.+?)\|(.+?)\|',
            line
        )
        if match:
            pattern_id = normalize_pattern_id(match.group(1))
            title = match.group(2)
            status = match.group(3)
            keywords_str = match.group(4)
            deps_str = match.group(5)

            # Extract keywords
            keywords = []
            kw_match = re.search(r'\*Keywords:\*\s*(.+?)(?:\*Queries|\.\s*$|$)', keywords_str)
            if kw_match:
                keywords = [k.strip() for k in kw_match.group(1).split(',') if k.strip()]

            # Extract dependencies
            dependencies = []
            for dep_pattern in [r'Builds on:\*?\*?\s*([^.]+)', r'Prerequisite for:\*?\*?\s*([^.]+)']:
                dep_match = re.search(dep_pattern, deps_str)
                if dep_match:
                    dependencies.extend([d.strip() for d in dep_match.group(1).split(',') if d.strip()])

            toc_sections[pattern_id] = {
                'title': title,
                'status': status,
                'keywords': keywords,
                'dependencies': dependencies,
            }

    return toc_sections


def get_section_content(
    spec_path: Path,
    sections: dict[str, FPFSection],
    pattern_id: str,
    max_chars: int = 15000,
) -> str | None:
    """
    Get content of a specific section.

    Returns None if section not found.
    """
    # Normalize the requested pattern ID
    normalized_id = normalize_pattern_id(pattern_id)

    if normalized_id not in sections:
        # Try partial match
        matches = [pid for pid in sections if pid.startswith(normalized_id)]
        if len(matches) == 1:
            normalized_id = matches[0]
        elif len(matches) > 1:
            return f"Ambiguous pattern ID '{pattern_id}'. Did you mean: {', '.join(sorted(matches))}?"
        else:
            # Try fuzzy match by searching
            fuzzy_matches = search_sections(sections, pattern_id)[:3]
            if fuzzy_matches:
                suggestions = [f"{s.pattern_id}: {s.title}" for s in fuzzy_matches]
                return f"Section '{pattern_id}' not found. Similar sections:\n" + '\n'.join(f"- {s}" for s in suggestions)
            return None

    section = sections[normalized_id]

    lines = spec_path.read_text(encoding='utf-8', errors='replace').split('\n')
    content_lines = lines[section.start_line:section.end_line + 1]
    content = '\n'.join(content_lines)

    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[Section truncated. Full section: {len(content)} chars]"

    return content


def search_sections(
    sections: dict[str, FPFSection],
    query: str,
    limit: int = 10,
) -> list[FPFSection]:
    """
    Search sections by keyword, title, or pattern ID.

    Returns matching sections sorted by relevance.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    results = []

    for section in sections.values():
        score = 0

        # Pattern ID exact match (highest priority)
        if query_lower == section.pattern_id.lower():
            score += 100

        # Pattern ID prefix match
        elif section.pattern_id.lower().startswith(query_lower):
            score += 50

        # Title contains query (case-insensitive)
        title_lower = section.title.lower()
        if query_lower in title_lower:
            # Bonus if query is at word boundary
            if re.search(rf'\b{re.escape(query_lower)}\b', title_lower):
                score += 40
            else:
                score += 30

        # Individual word matches in title
        title_words = set(title_lower.split())
        word_matches = len(query_words & title_words)
        if word_matches > 0:
            score += word_matches * 15

        # Keyword match
        for kw in section.keywords:
            kw_lower = kw.lower()
            if query_lower in kw_lower:
                score += 25
            elif any(w in kw_lower for w in query_words):
                score += 10

        # Part letter match (e.g., "B" matches all B.* sections)
        if len(query) == 1 and query.upper() == section.pattern_id[0]:
            score += 5

        if score > 0:
            results.append((score, section))

    results.sort(key=lambda x: (-x[0], x[1].pattern_id))
    return [s for _, s in results[:limit]]


def search_by_keywords(
    sections: dict[str, FPFSection],
    keywords: list[str],
    require_all: bool = False,
) -> list[FPFSection]:
    """
    Search sections by multiple keywords.

    Args:
        keywords: List of keywords to search for
        require_all: If True, all keywords must match. If False, any match counts.

    Returns matching sections sorted by match count.
    """
    keyword_lower = [k.lower() for k in keywords]
    results = []

    for section in sections.values():
        match_count = 0
        section_text = (
            section.title.lower() + ' ' +
            ' '.join(section.keywords).lower() + ' ' +
            section.pattern_id.lower()
        )

        for kw in keyword_lower:
            if kw in section_text:
                match_count += 1

        if require_all and match_count == len(keyword_lower):
            results.append((match_count, section))
        elif not require_all and match_count > 0:
            results.append((match_count, section))

    results.sort(key=lambda x: (-x[0], x[1].pattern_id))
    return [s for _, s in results]


def build_section_index_summary(sections: dict[str, FPFSection]) -> str:
    """
    Build a summary of all sections for the system prompt.
    """
    parts: dict[str, list[FPFSection]] = {}

    for section in sections.values():
        part = section.pattern_id[0]  # A, B, C, etc.
        if part not in parts:
            parts[part] = []
        parts[part].append(section)

    lines = ["## FPF Section Index\n"]
    lines.append("Use `read_fpf_section` action to retrieve full content.\n")

    part_names = {
        'A': 'Kernel Architecture (Foundational Ontology)',
        'B': 'Trans-disciplinary Reasoning (F-G-R, ADI, Evolution)',
        'C': 'Architheory Specifications (CAL/LOG/CHR)',
        'D': 'Ethics & Conflict Resolution',
        'E': 'Constitution & Authoring Guides',
        'F': 'Unification Suite (Bridges, Standards)',
        'G': 'SoTA Kit (AI/ML Integration)',
    }

    for part in sorted(parts.keys()):
        part_sections = sorted(parts[part], key=lambda s: s.pattern_id)
        part_name = part_names.get(part, 'Other')
        lines.append(f"\n### Part {part} — {part_name}")
        lines.append(f"({len(part_sections)} sections)\n")

        # Show more sections but keep it manageable
        for s in part_sections[:20]:
            status_marker = f" [{s.status}]" if s.status else ""
            lines.append(f"- **{s.pattern_id}**: {s.title}{status_marker}")

        if len(part_sections) > 20:
            lines.append(f"  ... and {len(part_sections) - 20} more sections")

    return '\n'.join(lines)


def get_sections_by_part(sections: dict[str, FPFSection], part: str) -> list[FPFSection]:
    """Get all sections in a specific part (A, B, C, etc.)."""
    part_upper = part.upper()
    return sorted(
        [s for s in sections.values() if s.pattern_id.startswith(part_upper)],
        key=lambda s: s.pattern_id
    )


def get_related_sections(
    sections: dict[str, FPFSection],
    pattern_id: str,
) -> list[FPFSection]:
    """
    Get sections related to the given pattern ID.

    Returns sections that:
    - Are listed as dependencies of this section
    - Have this section as a dependency
    - Are siblings (same parent, e.g., A.1.1 and A.1.2 for A.1)
    """
    normalized_id = normalize_pattern_id(pattern_id)
    if normalized_id not in sections:
        return []

    section = sections[normalized_id]
    related = set()

    # Add explicit dependencies
    for dep in section.dependencies:
        dep_normalized = normalize_pattern_id(dep.strip())
        if dep_normalized in sections:
            related.add(dep_normalized)

    # Find sections that depend on this one
    for other_id, other_section in sections.items():
        for dep in other_section.dependencies:
            if normalize_pattern_id(dep.strip()) == normalized_id:
                related.add(other_id)

    # Add siblings (same parent)
    parts = normalized_id.split('.')
    if len(parts) >= 2:
        parent = '.'.join(parts[:-1])
        for other_id in sections:
            if other_id.startswith(parent + '.') and other_id != normalized_id:
                related.add(other_id)

    return sorted([sections[pid] for pid in related], key=lambda s: s.pattern_id)
