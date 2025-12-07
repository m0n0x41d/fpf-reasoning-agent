"""
FPF Specification Index - Parse and index FPF spec for section retrieval.

Pure functions for building and querying the FPF index.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FPFSection:
    """A section in the FPF specification."""
    pattern_id: str
    title: str
    status: str
    start_line: int
    end_line: int
    keywords: list[str]
    dependencies: list[str]


# Pattern to match section headers like "## A.1 Holonic Foundation" or "### Pattern A.1.1"
SECTION_HEADER_PATTERNS = [
    # "## A.1 Holonic Foundation..."
    re.compile(r'^##\s+(?:Pattern\s+)?([A-Z]\.\d+(?:\.\d+)?)\s*[—–-]?\s*(.+)$'),
    # "### Pattern A.1.1 — Title"
    re.compile(r'^###\s+(?:\*\*)?(?:Pattern\s+)?([A-Z]\.\d+(?:\.\d+)?)\s*[—–-]?\s*(.+?)(?:\*\*)?(?:\s*\[.\])?$'),
    # "| A.1 | **Holonic Foundation..." in TOC
    re.compile(r'^\|\s*([A-Z]\.\d+(?:\.\d+)?)\s*\|\s*\*\*(.+?)\*\*'),
]

# Pattern to extract Part headers
PART_PATTERN = re.compile(r'^#\s*Part\s+([A-Z])\s*[—–·-]\s*(.+)$', re.IGNORECASE)

# Alternative: look for explicit pattern definitions
PATTERN_DEF = re.compile(r'^##\s+([A-Z]\.\d+)\s+(.+?)\s+\\\[([A-Z])\\\]')


def parse_fpf_spec(spec_path: Path) -> dict[str, FPFSection]:
    """
    Parse FPF specification and build section index.

    Returns dict mapping pattern_id to FPFSection.
    """
    if not spec_path.exists():
        return {}

    lines = spec_path.read_text(encoding='utf-8', errors='replace').split('\n')

    sections: dict[str, FPFSection] = {}
    current_part = ""

    # First pass: find all section starts from TOC
    toc_sections = _extract_toc_sections(lines)

    # Second pass: find actual section locations in body
    for i, line in enumerate(lines):
        # Track current part
        part_match = PART_PATTERN.match(line)
        if part_match:
            current_part = part_match.group(1)
            continue

        # Look for section headers
        for pattern in SECTION_HEADER_PATTERNS:
            match = pattern.match(line)
            if match:
                pattern_id = match.group(1)
                title = match.group(2).strip()

                # Clean up title
                title = re.sub(r'\*\*', '', title)
                title = re.sub(r'\s*\[.\]\s*$', '', title)
                title = title.strip('` ')

                if pattern_id not in sections:
                    sections[pattern_id] = FPFSection(
                        pattern_id=pattern_id,
                        title=title,
                        status="",
                        start_line=i,
                        end_line=i,  # Will be updated
                        keywords=[],
                        dependencies=[],
                    )
                else:
                    # Update start line if this is the actual section (not TOC)
                    if i > 200:  # After TOC
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
        if pid in sections:
            sections[pid].keywords = toc_info.get('keywords', [])
            sections[pid].status = toc_info.get('status', '')
            sections[pid].dependencies = toc_info.get('dependencies', [])

    return sections


def _extract_toc_sections(lines: list[str]) -> dict[str, dict]:
    """Extract section info from Table of Contents."""
    toc_sections = {}

    for line in lines[:500]:  # TOC is in first 500 lines
        # Match TOC rows like: | A.1 | **Title** | Stable | keywords... | deps... |
        match = re.match(
            r'^\|\s*([A-Z]\.\d+(?:\.\d+)?)\s*\|\s*\*\*(.+?)\*\*\s*\|\s*(\w+)\s*\|(.+?)\|(.+?)\|',
            line
        )
        if match:
            pattern_id = match.group(1)
            title = match.group(2)
            status = match.group(3)
            keywords_str = match.group(4)
            deps_str = match.group(5)

            # Extract keywords
            keywords = []
            kw_match = re.search(r'\*Keywords:\*\s*(.+?)(?:\*Queries|\.|$)', keywords_str)
            if kw_match:
                keywords = [k.strip() for k in kw_match.group(1).split(',')]

            # Extract dependencies
            dependencies = []
            for dep_pattern in [r'Builds on:\*?\*?\s*([^.]+)', r'Prerequisite for:\*?\*?\s*([^.]+)']:
                dep_match = re.search(dep_pattern, deps_str)
                if dep_match:
                    dependencies.extend([d.strip() for d in dep_match.group(1).split(',')])

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
    if pattern_id not in sections:
        # Try partial match
        matches = [pid for pid in sections if pid.startswith(pattern_id)]
        if len(matches) == 1:
            pattern_id = matches[0]
        elif len(matches) > 1:
            return f"Ambiguous pattern ID '{pattern_id}'. Did you mean: {', '.join(matches)}?"
        else:
            return None

    section = sections[pattern_id]

    lines = spec_path.read_text(encoding='utf-8', errors='replace').split('\n')
    content_lines = lines[section.start_line:section.end_line + 1]
    content = '\n'.join(content_lines)

    if len(content) > max_chars:
        content = content[:max_chars] + f"\n\n[Section truncated. Total: {len(content)} chars]"

    return content


def search_sections(
    sections: dict[str, FPFSection],
    query: str,
) -> list[FPFSection]:
    """
    Search sections by keyword or title.

    Returns matching sections sorted by relevance.
    """
    query_lower = query.lower()
    results = []

    for section in sections.values():
        score = 0

        # Pattern ID exact match
        if query_lower == section.pattern_id.lower():
            score += 100

        # Pattern ID prefix
        if section.pattern_id.lower().startswith(query_lower):
            score += 50

        # Title match
        if query_lower in section.title.lower():
            score += 30

        # Keyword match
        for kw in section.keywords:
            if query_lower in kw.lower():
                score += 20

        if score > 0:
            results.append((score, section))

    results.sort(key=lambda x: -x[0])
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

    part_names = {
        'A': 'Kernel Architecture',
        'B': 'Trans-disciplinary Reasoning',
        'C': 'Architheory Specifications',
        'D': 'Ethics & Conflict',
        'E': 'Constitution & Authoring',
        'F': 'Unification Suite',
        'G': 'SoTA Kit',
    }

    for part in sorted(parts.keys()):
        part_sections = sorted(parts[part], key=lambda s: s.pattern_id)
        lines.append(f"\n### Part {part} — {part_names.get(part, 'Other')}\n")

        for s in part_sections[:15]:  # Limit to avoid huge index
            lines.append(f"- **{s.pattern_id}**: {s.title}")

    return '\n'.join(lines)
