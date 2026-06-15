#!/usr/bin/env python3
"""Render Obsidian-flavoured RL tutorial Markdown into A4-print-friendly HTML.

Features:
  * YAML frontmatter -> document <header class="meta">
  * KaTeX inline ($...$) and display ($$...$$), preserved verbatim for client KaTeX auto-render
  * Obsidian callouts (> [!type] title / body), with nested markdown rendered correctly
  * Mermaid (```mermaid) and Graphviz (```dot) code blocks rendered client-side
  * highlight.js syntax highlighting for code
  * Wikilinks [[X]] and [[X|alias]] rendered as <span class="wikilink">
  * GFM tables, task lists, footnote-free body
  * Shared stylesheet at html_export/_assets/style.css with @media print A4 rules

Usage:
    uv run python tools/render_md_to_html.py PPO教程.md
    uv run python tools/render_md_to_html.py --all
    uv run python tools/render_md_to_html.py --all --out-dir html_export
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path
from typing import Iterable

import yaml
from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.tasklists import tasklists_plugin


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CALLOUT_META = {
    "abstract":  ("📋", "#7c3aed"),
    "summary":   ("📌", "#64748b"),
    "tldr":      ("📋", "#7c3aed"),
    "note":      ("📝", "#3b82f6"),
    "info":      ("ℹ️", "#06b6d4"),
    "todo":      ("📝", "#3b82f6"),
    "tip":       ("💡", "#10b981"),
    "hint":      ("💡", "#10b981"),
    "important": ("💡", "#10b981"),
    "success":   ("✨", "#10b981"),
    "check":     ("✅", "#10b981"),
    "done":      ("✅", "#10b981"),
    "checklist": ("✅", "#10b981"),
    "question":  ("❓", "#06b6d4"),
    "help":      ("❓", "#06b6d4"),
    "faq":       ("❓", "#06b6d4"),
    "warning":   ("⚠️", "#f59e0b"),
    "caution":   ("⚠️", "#f59e0b"),
    "attention": ("⚠️", "#f59e0b"),
    "danger":    ("🔥", "#ef4444"),
    "error":     ("🔥", "#ef4444"),
    "bug":       ("🐛", "#ef4444"),
    "example":   ("🔬", "#6366f1"),
    "quote":     ("❝", "#64748b"),
    "cite":      ("❝", "#64748b"),
}

DEFAULT_OUT_DIR = "html_export"
SHARED_CSS_REL = "_assets/style.css"

CALLOUT_PLACEHOLDER_FMT = "@@CALLOUT_PLACEHOLDER_{idx}@@"
WIKILINK_RE = re.compile(r"\[\[([^\[\]\n|]+?)(?:\|([^\[\]\n]+?))?\]\]")


# Obsidian writes block-level math `$$` flush against surrounding prose. The
# commonmark spec (and dollarmath_plugin's block detector) require a blank line
# above and below, otherwise it is parsed as inline math, breaking on the
# embedded newlines. Pre-process to insert blank lines around any line that is
# *exactly* `$$` (optionally followed by inline content as in `$$ foo $$`).
DOLLAR_BLOCK_LINE_RE = re.compile(r"^[ \t]*\$\$[ \t]*$", re.MULTILINE)


def normalise_block_math(text: str) -> str:
    """Insert blank lines around standalone `$$` markers so dollarmath_plugin
    recognises them as block math even when authors omit blank lines (Obsidian
    style). Only touches lines outside fenced code blocks."""
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # Track fenced code blocks (``` or ~~~ with the same opener).
        if not in_fence:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = True
                fence_marker = "```" if stripped.startswith("```") else "~~~"
                out.append(line)
                continue
        else:
            out.append(line)
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue
        # Outside any code fence: a bare `$$` line needs blank-line padding.
        if line.strip() == "$$":
            if out and out[-1].strip() != "":
                out.append("")
            out.append(line)
            # Look ahead: if the next line is non-blank and is not also `$$`,
            # we still want the *closing* `$$` to be padded below; simplest is
            # to also pad after every standalone `$$`. The next iteration will
            # re-examine the next line and we won't add a duplicate blank.
            if i + 1 < len(lines) and lines[i + 1].strip() != "":
                out.append("")
            continue
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Frontmatter handling
# ---------------------------------------------------------------------------

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter dict, remaining body). Empty dict on parse failure."""
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    try:
        meta = yaml.safe_load(m.group(1)) or {}
        if not isinstance(meta, dict):
            meta = {}
    except yaml.YAMLError:
        meta = {}
    body = text[m.end():]
    return meta, body


def render_meta_header(meta: dict, fallback_title: str) -> str:
    title = html.escape(str(meta.get("title") or fallback_title))
    tags = meta.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    tag_html = "".join(
        f'<span class="meta-tag">#{html.escape(str(t))}</span>'
        for t in tags
    )
    source = meta.get("source")
    source_html = ""
    if source:
        s = html.escape(str(source))
        source_html = f'<span class="meta-source">来源：<a href="{s}">{s}</a></span>'
    created = meta.get("created")
    created_html = (
        f'<span class="meta-date">{html.escape(str(created))}</span>'
        if created else ""
    )
    return (
        '<header class="meta">\n'
        f'  <h1 class="doc-title">{title}</h1>\n'
        '  <div class="meta-row">\n'
        f'    <span class="meta-tags">{tag_html}</span>\n'
        f'    {source_html}\n'
        f'    {created_html}\n'
        '  </div>\n'
        '</header>\n'
    )


# ---------------------------------------------------------------------------
# Obsidian callout extraction
# ---------------------------------------------------------------------------

CALLOUT_HEADER_RE = re.compile(
    r"^>\s*\[!(?P<type>[A-Za-z][\w-]*)\](?P<flag>[+-]?)\s*(?P<title>.*?)\s*$"
)


def _strip_quote_prefix(line: str) -> str:
    """Strip a single leading `> ` (or `>`) from a callout-body line."""
    if line.startswith("> "):
        return line[2:]
    if line.startswith(">"):
        return line[1:]
    return line


def extract_callouts(body: str, child_md: MarkdownIt) -> tuple[str, list[str]]:
    """Replace each callout block with a placeholder, return (new_body, [callout_html...]).

    A callout is:
        > [!type] optional title
        > body line
        > body line
        ...
    Termination: first line that does NOT start with `>` (typically a blank line)
    or end-of-file. Lines starting with `>` (including `>` alone) are part of
    the callout body and survive as paragraph breaks.
    """
    lines = body.split("\n")
    out_lines: list[str] = []
    callouts: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = CALLOUT_HEADER_RE.match(line)
        if not m:
            out_lines.append(line)
            i += 1
            continue
        c_type = m.group("type").lower()
        c_title = m.group("title") or ""
        body_lines: list[str] = []
        j = i + 1
        while j < n and lines[j].startswith(">"):
            body_lines.append(_strip_quote_prefix(lines[j]))
            j += 1
        inner_md_text = "\n".join(body_lines).rstrip()
        inner_md_text = normalise_block_math(inner_md_text)
        # Render the inner markdown with the *same* engine (recursively
        # supporting callouts is unnecessary here; Obsidian doesn't nest them
        # in this corpus, but if they did appear we'd just see them as raw
        # blockquotes — acceptable fallback).
        inner_html = child_md.render(inner_md_text) if inner_md_text else ""
        emoji, color = CALLOUT_META.get(c_type, ("📌", "#64748b"))
        title_html = child_md.renderInline(c_title) if c_title else ""
        callout_html = (
            f'<div class="callout callout-{c_type}" style="border-left-color:{color};'
            f' background:{color}14;">\n'
            f'  <div class="callout-title"><span class="callout-icon">{emoji}</span>'
            f'<span class="callout-title-text">{title_html}</span></div>\n'
            f'  <div class="callout-body">{inner_html}</div>\n'
            f'</div>'
        )
        idx = len(callouts)
        callouts.append(callout_html)
        # Surround the placeholder with blank lines so markdown-it treats it as
        # an isolated HTML block (with html=True it will pass through).
        out_lines.append("")
        out_lines.append(CALLOUT_PLACEHOLDER_FMT.format(idx=idx))
        out_lines.append("")
        i = j
    return "\n".join(out_lines), callouts


# ---------------------------------------------------------------------------
# Wikilinks
# ---------------------------------------------------------------------------

def _wikilink_sub(match: re.Match) -> str:
    target = match.group(1).strip()
    alias = match.group(2)
    label = (alias or target).strip()
    label_esc = html.escape(label)
    target_esc = html.escape(target)
    return (
        f'<span class="wikilink" data-target="{target_esc}" '
        f'title="{target_esc}">{label_esc}</span>'
    )


def replace_wikilinks(text: str) -> str:
    return WIKILINK_RE.sub(_wikilink_sub, text)


# ---------------------------------------------------------------------------
# Markdown-it setup
# ---------------------------------------------------------------------------

def _math_inline_renderer(self, tokens, idx, options, env):
    # Re-emit as raw $...$ for client-side KaTeX auto-render.
    # CRITICAL: escape `<` `>` `&` so the browser does not parse e.g. `o_{<t}`
    # as the start of an HTML tag. KaTeX auto-render reads textContent, which
    # decodes &lt; back to `<` before passing to KaTeX, so math is unaffected.
    content = html.escape(tokens[idx].content, quote=False)
    return f"<span class=\"math math-inline\">${content}$</span>"


def _math_block_renderer(self, tokens, idx, options, env):
    content = html.escape(tokens[idx].content.strip("\n"), quote=False)
    label = tokens[idx].info  # if dollarmath labels enabled
    label_attr = f' id="eq-{html.escape(label)}"' if label else ""
    return f"<div class=\"math math-block\"{label_attr}>\n$$\n{content}\n$$\n</div>\n"


def _fence_renderer(self, tokens, idx, options, env):
    """Custom fence renderer:
       * mermaid -> <pre class="mermaid">...</pre>
       * dot     -> <pre><code class="language-dot">...</code></pre> (replaced client-side by viz.js)
       * other   -> <pre><code class="language-X">...escaped...</code></pre> for highlight.js
    """
    token = tokens[idx]
    info = (token.info or "").strip()
    lang = info.split(None, 1)[0] if info else ""
    content = token.content
    if lang == "mermaid":
        return f'<pre class="mermaid">{html.escape(content)}</pre>\n'
    cls = f' class="language-{html.escape(lang)}"' if lang else ""
    return f'<pre><code{cls}>{html.escape(content)}</code></pre>\n'


def _link_open_renderer(self, tokens, idx, options, env):
    # Tag external links to open in a new tab; defer actual emission to default.
    token = tokens[idx]
    href = token.attrGet("href") or ""
    if href.startswith(("http://", "https://")):
        token.attrSet("target", "_blank")
        token.attrSet("rel", "noopener noreferrer")
    return self.renderToken(tokens, idx, options, env)


def build_md() -> tuple[MarkdownIt, MarkdownIt]:
    """Return (outer_md, inner_md). Inner is used for callout body rendering."""
    def make() -> MarkdownIt:
        md = (
            MarkdownIt("commonmark", {"html": True, "linkify": True, "breaks": False})
            .enable("table")
            .enable("strikethrough")
            .use(dollarmath_plugin, allow_labels=True, allow_space=True,
                 allow_digits=True, double_inline=True)
            .use(tasklists_plugin, enabled=True)
            .use(anchors_plugin, max_level=4, slug_func=lambda s: re.sub(r"\s+", "-", s.strip()))
        )
        md.add_render_rule("math_inline", _math_inline_renderer)
        md.add_render_rule("math_inline_double", _math_inline_renderer)
        md.add_render_rule("math_block", _math_block_renderer)
        md.add_render_rule("math_block_label", _math_block_renderer)
        md.add_render_rule("fence", _fence_renderer)
        md.add_render_rule("link_open", _link_open_renderer)
        return md
    return make(), make()


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="{css_href}">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/styles/github.min.css">
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/@viz-js/viz@3/lib/viz-standalone.js"></script>
  <script src="https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11/build/highlight.min.js"></script>
  <script>
    document.addEventListener('DOMContentLoaded', async function () {{
      // 1. highlight.js — only on blocks that opted in via language-XXX class
      try {{
        document.querySelectorAll('pre code').forEach(function (block) {{
          if (block.classList.contains('language-dot')) return;
          if (block.classList.contains('language-mermaid')) return;
          // Skip auto-detect on un-tagged blocks (e.g. ASCII diagrams) — hljs
          // mis-classifies short text and recolours it badly.
          var hasLang = false;
          for (var i = 0; i < block.classList.length; i++) {{
            if (block.classList[i].indexOf('language-') === 0) {{ hasLang = true; break; }}
          }}
          if (!hasLang) return;
          window.hljs && window.hljs.highlightElement(block);
        }});
      }} catch (e) {{ console.warn('hljs failed', e); }}

      // 2. Graphviz: replace <pre><code class="language-dot"> with rendered SVG
      try {{
        if (window.Viz && typeof window.Viz.instance === 'function') {{
          const vizInstance = await window.Viz.instance();
          document.querySelectorAll('pre code.language-dot').forEach(function (block) {{
            try {{
              const svg = vizInstance.renderSVGElement(block.textContent);
              const wrapper = document.createElement('div');
              wrapper.className = 'graphviz';
              wrapper.appendChild(svg);
              block.parentElement.replaceWith(wrapper);
            }} catch (err) {{
              console.error('viz render failed', err);
              block.parentElement.classList.add('graphviz-error');
            }}
          }});
        }} else {{
          console.warn('viz-js not loaded');
        }}
      }} catch (e) {{ console.warn('viz init failed', e); }}

      // 3. Mermaid
      try {{
        if (window.mermaid) {{
          window.mermaid.initialize({{ startOnLoad: false, theme: 'default' }});
          await window.mermaid.run({{ querySelector: 'pre.mermaid' }});
        }}
      }} catch (e) {{ console.warn('mermaid failed', e); }}

      // 4. KaTeX last so it does not fight hljs/viz
      try {{
        if (window.renderMathInElement) {{
          window.renderMathInElement(document.body, {{
            delimiters: [
              {{ left: '$$', right: '$$', display: true }},
              {{ left: '$',  right: '$',  display: false }}
            ],
            throwOnError: false,
            ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
          }});
        }}
      }} catch (e) {{ console.warn('katex failed', e); }}
    }});
  </script>
</head>
<body>
{header}
<main class="doc">
{body}
</main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def render_file(md_path: Path, out_dir: Path) -> Path:
    text = md_path.read_text(encoding="utf-8")
    meta, body = split_frontmatter(text)

    # Pad standalone `$$` lines with blank lines so block math is recognised
    # even when the source omits blank lines (Obsidian convention).
    body = normalise_block_math(body)

    outer_md, inner_md = build_md()

    # 1. Extract callouts BEFORE wikilinks substitution so the inner markdown
    #    still contains [[...]] which we'll process in two passes.
    body_with_holders, callouts = extract_callouts(body, inner_md)

    # 2. Wikilinks: do them on raw markdown — they are inline tokens that
    #    survive markdown-it (we surface them as inline HTML; html=True keeps
    #    them as raw HTML during inline parsing).
    body_with_holders = replace_wikilinks(body_with_holders)
    callouts = [replace_wikilinks(c) for c in callouts]

    # 3. Render markdown.
    body_html = outer_md.render(body_with_holders)

    # 4. Sub callout placeholders back in.
    for idx, callout_html in enumerate(callouts):
        placeholder = CALLOUT_PLACEHOLDER_FMT.format(idx=idx)
        # markdown-it may have wrapped a lone placeholder line in <p>...</p>
        # depending on surrounding blanks. Handle both cases.
        wrapped = f"<p>{placeholder}</p>"
        if wrapped in body_html:
            body_html = body_html.replace(wrapped, callout_html)
        else:
            body_html = body_html.replace(placeholder, callout_html)

    # 5. Build header/title.
    fallback_title = md_path.stem
    header_html = render_meta_header(meta, fallback_title)
    title = str(meta.get("title") or fallback_title)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "_assets").mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{md_path.stem}.html"
    out_path.write_text(
        HTML_TEMPLATE.format(
            title=html.escape(title),
            css_href=SHARED_CSS_REL,
            header=header_html,
            body=body_html,
        ),
        encoding="utf-8",
    )
    return out_path


CSS_CONTENT = r"""/* ============================================================
   RL Tutorial — A4 print-friendly stylesheet
   ============================================================ */
:root {
  --fg: #1f2328;
  --fg-soft: #57606a;
  --bg: #ffffff;
  --bg-soft: #f8fafc;
  --border: #d0d7de;
  --border-soft: #e1e4e8;
  --code-bg: #f6f8fa;
  --inline-code-bg: #eef0f2;
  --link: #0969da;
  --accent: #6366f1;
}

html, body {
  margin: 0;
  padding: 0;
  background: var(--bg-soft);
  color: var(--fg);
  font-family: -apple-system, BlinkMacSystemFont, "PingFang SC",
    "Helvetica Neue", "Microsoft YaHei", "Hiragino Sans GB", sans-serif;
  font-size: 16px;
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
}

body {
  max-width: 820px;
  margin: 0 auto;
  padding: 32px 28px 64px;
  background: var(--bg);
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}

main.doc {
  word-wrap: break-word;
  overflow-wrap: anywhere;
}

/* ---------- Header / metadata ---------- */
header.meta {
  border-bottom: 1px solid var(--border);
  padding-bottom: 14px;
  margin-bottom: 24px;
}
header.meta h1.doc-title {
  font-size: 1.85em;
  margin: 0 0 10px 0;
  line-height: 1.25;
  color: #111;
  page-break-after: avoid;
}
.meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  font-size: 0.85em;
  color: var(--fg-soft);
  align-items: center;
}
.meta-tags { display: inline-flex; flex-wrap: wrap; gap: 4px; }
.meta-tag {
  background: #eef2f6;
  color: #475569;
  padding: 1px 8px;
  border-radius: 999px;
  font-size: 0.85em;
}
.meta-source a { color: var(--link); text-decoration: none; }
.meta-source a:hover { text-decoration: underline; }
.meta-date { color: var(--fg-soft); }

/* ---------- Headings ---------- */
h1, h2, h3, h4, h5, h6 {
  margin: 1.6em 0 0.6em;
  line-height: 1.3;
  font-weight: 650;
  page-break-after: avoid;
}
h1 { font-size: 1.75em; border-bottom: 2px solid var(--border); padding-bottom: 0.25em; }
h2 { font-size: 1.4em;  border-bottom: 1px solid var(--border); padding-bottom: 0.2em; }
h3 { font-size: 1.18em; }
h4 { font-size: 1.05em; }

/* ---------- Body ---------- */
p { margin: 0.6em 0; }
ul, ol { margin: 0.5em 0; padding-left: 1.6em; }
li { margin: 0.2em 0; }
li > p { margin: 0.2em 0; }
hr { border: none; border-top: 1px solid var(--border); margin: 1.6em 0; }

a { color: var(--link); text-decoration: none; }
a:hover { text-decoration: underline; }

/* ---------- Inline code ---------- */
code {
  font-family: "SF Mono", "JetBrains Mono", Menlo, Consolas,
    "Liberation Mono", "Source Code Pro", monospace;
  background: var(--inline-code-bg);
  font-size: 0.88em;
  padding: 0.1em 0.35em;
  border-radius: 3px;
}

/* ---------- Code blocks ---------- */
pre {
  background: var(--code-bg);
  border: 1px solid var(--border-soft);
  border-radius: 6px;
  padding: 12px 14px;
  overflow: auto;
  font-size: 0.86em;
  line-height: 1.5;
  page-break-inside: avoid;
}
pre code {
  background: transparent;
  padding: 0;
  border-radius: 0;
  font-size: 1em;
  white-space: pre;
}

/* ---------- Tables ---------- */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
  font-size: 0.93em;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid var(--border);
  padding: 6px 10px;
  text-align: left;
  vertical-align: top;
}
thead th {
  background: #f1f5f9;
  font-weight: 600;
}
tbody tr:nth-child(even) td { background: #fafbfc; }

/* ---------- Blockquote ---------- */
blockquote {
  border-left: 4px solid var(--border);
  margin: 1em 0;
  padding: 0.4em 1em;
  color: var(--fg-soft);
  background: #fafbfc;
  page-break-inside: avoid;
}

/* ---------- Callouts (Obsidian) ---------- */
.callout {
  border-left: 4px solid var(--accent);
  border-radius: 6px;
  padding: 10px 14px 10px 14px;
  margin: 1em 0;
  background: rgba(99,102,241,0.08);
  page-break-inside: avoid;
}
.callout > .callout-title {
  font-weight: 650;
  display: flex;
  gap: 6px;
  align-items: center;
  margin-bottom: 4px;
  color: #1f2328;
}
.callout .callout-icon { font-size: 1em; }
.callout .callout-body > :first-child { margin-top: 0; }
.callout .callout-body > :last-child  { margin-bottom: 0; }
.callout pre { background: rgba(255,255,255,0.7); }
.callout code { background: rgba(255,255,255,0.7); }

/* Per-type accent colours (kept here so screen view matches print) */
.callout-abstract  { border-left-color: #7c3aed; }
.callout-summary   { border-left-color: #64748b; }
.callout-tldr      { border-left-color: #7c3aed; }
.callout-note      { border-left-color: #3b82f6; }
.callout-info      { border-left-color: #06b6d4; }
.callout-tip       { border-left-color: #10b981; }
.callout-hint      { border-left-color: #10b981; }
.callout-important { border-left-color: #10b981; }
.callout-success   { border-left-color: #10b981; }
.callout-check     { border-left-color: #10b981; }
.callout-done      { border-left-color: #10b981; }
.callout-checklist { border-left-color: #10b981; }
.callout-question  { border-left-color: #06b6d4; }
.callout-help      { border-left-color: #06b6d4; }
.callout-faq       { border-left-color: #06b6d4; }
.callout-warning   { border-left-color: #f59e0b; }
.callout-caution   { border-left-color: #f59e0b; }
.callout-attention { border-left-color: #f59e0b; }
.callout-danger    { border-left-color: #ef4444; }
.callout-error     { border-left-color: #ef4444; }
.callout-bug       { border-left-color: #ef4444; }
.callout-example   { border-left-color: #6366f1; }
.callout-quote     { border-left-color: #64748b; }
.callout-cite      { border-left-color: #64748b; }

/* ---------- Wiki links ---------- */
.wikilink {
  background: #eef2f6;
  color: #475569;
  border-radius: 3px;
  padding: 0 0.32em;
  font-size: 0.95em;
  text-decoration: none;
}

/* ---------- Math ---------- */
.math.math-block {
  text-align: center;
  margin: 1em 0;
  page-break-inside: avoid;
  overflow-x: auto;
}
.katex-display {
  margin: 0.6em 0 !important;
  page-break-inside: avoid;
}
.katex { font-size: 1.05em; }

/* ---------- Graphviz / Mermaid ---------- */
.graphviz, pre.mermaid {
  display: flex;
  justify-content: center;
  padding: 1em 0;
  page-break-inside: avoid;
}
/* SVG 自身可能带 width/height 属性，必须 !important 覆盖；本地生成的图（graphviz/mermaid）限到 90% */
.graphviz svg,
pre.mermaid svg {
  width: 90% !important;
  max-width: 90% !important;
  height: auto !important;
}
.graphviz-error { border: 1px dashed #ef4444; }
pre.mermaid { background: transparent; border: none; }

/* markdown 里 ![](url) 引入的外部图片限到 65%，避免 datawhale 等大尺寸 PNG 占满整页 */
img { display: block; margin: 0 auto; max-width: 65%; height: auto; }

/* ---------- Task lists ---------- */
ul.contains-task-list { list-style: none; padding-left: 1em; }
li.task-list-item input[type=checkbox] { margin-right: 0.4em; }

/* ---------- Print (A4) ---------- */
@page {
  size: A4;
  margin: 1.5cm 1.5cm 2cm 1.5cm;
}

@media print {
  html, body {
    background: #ffffff;
    font-size: 11pt;
    line-height: 1.65;
    color: #000;
  }
  body {
    max-width: none;
    margin: 0;
    padding: 0;
    box-shadow: none;
  }
  header.meta {
    border-bottom: 1px solid #999;
    margin-bottom: 1em;
    padding-bottom: 0.5em;
  }
  header.meta h1.doc-title { font-size: 22pt; }
  h1 { font-size: 22pt; border-color: #999; }
  h2 {
    font-size: 16pt;
    border-color: #999;
    page-break-before: auto;
    page-break-after: avoid;
  }
  h3 { font-size: 13pt; }
  h4 { font-size: 11.5pt; }
  p, li { orphans: 3; widows: 3; }

  pre, pre code {
    font-size: 9.5pt;
    line-height: 1.4;
  }
  pre {
    border-color: #bbb;
    page-break-inside: avoid;
  }

  table { font-size: 9.5pt; page-break-inside: avoid; }
  thead th { background: #eee !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  tbody tr:nth-child(even) td { background: #f7f7f7 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }

  blockquote, .callout { page-break-inside: avoid; }
  .callout { -webkit-print-color-adjust: exact; print-color-adjust: exact; }

  .math.math-block, .katex-display, .graphviz, pre.mermaid {
    page-break-inside: avoid;
  }

  /* Do not append URL after links when printing */
  a, a:visited { color: #000; text-decoration: none; }
  a[href]::after { content: ""; }

  .wikilink { background: #eee !important; color: #333 !important;
              -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
"""


def write_css(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    css_dir = out_dir / "_assets"
    css_dir.mkdir(parents=True, exist_ok=True)
    css_path = css_dir / "style.css"
    css_path.write_text(CSS_CONTENT, encoding="utf-8")
    return css_path


def discover_md(project_root: Path) -> list[Path]:
    return sorted(
        p for p in project_root.iterdir()
        if p.is_file() and p.suffix == ".md" and not p.name.startswith("_")
    )


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("inputs", nargs="*", help="markdown files to render")
    ap.add_argument("--all", action="store_true",
                    help="render all *.md in the project root (excluding files starting with _)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help=f"output directory (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--project-root",
                    default=str(Path(__file__).resolve().parent.parent),
                    help="project root for --all (default: repo root)")
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()

    if args.all:
        files = discover_md(project_root)
    else:
        if not args.inputs:
            ap.error("provide one or more markdown files, or pass --all")
        files = []
        for s in args.inputs:
            p = Path(s)
            if not p.is_absolute():
                p = (project_root / p).resolve()
            if not p.exists():
                print(f"  ! missing: {p}", file=sys.stderr)
                continue
            files.append(p)

    if not files:
        print("nothing to render", file=sys.stderr)
        return 1

    css_path = write_css(out_dir)
    print(f"  css -> {css_path}")

    rc = 0
    for md_path in files:
        try:
            out_path = render_file(md_path, out_dir)
            print(f"  md  -> {out_path}")
        except Exception as e:
            print(f"  ! failed {md_path}: {e}", file=sys.stderr)
            rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
