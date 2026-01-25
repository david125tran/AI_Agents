# ./renderers/report_pdf.py

# ---------------------------------- Imports ----------------------------------
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import re



# ---------------------------------- Save Report as PDF ----------------------------------
def save_report_pdf(out: Dict[str, Any], *, script_dir: Path) -> Optional[Path]:
    """
    Modern-ish, “stunning” PDF renderer for the stock due diligence report.

    Philosophy:
      - Make the first page feel like a real report (title, subtitle, key stats).
      - Keep body typography clean and readable (good spacing, subtle hierarchy).
      - Render the deterministic snapshot as a compact “facts card”.
      - Support the markdown patterns your advisor emits:
          • Headings: ## / ### (and # if it shows up)
          • Bullets: - item
          • Numbered: 1. item
          • Horizontal rules: --- / ***
          • Inline bold: **text**
          • Source links: [[Source: label]](url)  (already produced by replace_citations)
      - If something is missing, fail gracefully (don’t crash a long run at the finish line).

    Returns:
        Path to the generated PDF, or None if there's no report markdown.
    """
    # --- local imports so this function is portable if you move it to a utils file ---
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        PageBreak,
        Table,
        TableStyle,
        ListFlowable,
        ListItem,
        HRFlowable,
        KeepTogether,
    )
    from reportlab.pdfbase.pdfmetrics import stringWidth

    # -----------------------------
    # 1) Pull the content we need
    # -----------------------------
    report = out.get("report") or {}
    md = (report.get("report_markdown") or "").strip()
    if not md:
        return None

    ticker = (out.get("ticker") or "UNKNOWN").strip().upper()
    ts_utc = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    generated_human = datetime.now(timezone.utc).strftime("%b %d, %Y • %H:%M UTC")

    reports_dir = script_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = reports_dir / f"{ticker}_due_diligence_{ts_utc}.pdf"

    # -----------------------------
    # 2) Design tokens (theme)
    # -----------------------------
    # Keep it subtle. Dark text, a single accent, soft neutrals.
    INK = HexColor("#0B1220")          # near-black, easier on eyes than pure black
    MUTED = HexColor("#4B5563")        # gray
    SOFT = HexColor("#E5E7EB")         # light gray borders
    ACCENT = HexColor("#2563EB")       # blue accent
    ACCENT_DARK = HexColor("#1E40AF")  # darker blue (headers)
    PANEL_BG = HexColor("#F8FAFC")     # almost-white panel background

    # -----------------------------
    # 3) Helpers (escaping + links)
    # -----------------------------
    def _escape_rl(s: str) -> str:
        """
        ReportLab Paragraph uses a mini-HTML parser.
        Escape XML-ish characters so arbitrary text doesn't break rendering.
        """
        return (
            (s or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _md_inline_to_rl(s: str) -> str:
        """
        Convert the small subset of inline markdown we expect into ReportLab tags.
        - **bold**
        - [[Source: label]](url) -> hyperlink
        """
        s = _escape_rl(s)

        # Convert [[Source: ...]](http...) into <a href="...">Source: ...</a>
        def link_repl(m: re.Match) -> str:
            label = (m.group(1) or "").strip()
            url = (m.group(2) or "").strip()
            label = _escape_rl(label)
            # A tiny style trick: make links slightly accent-colored.
            return f'<a href="{url}" color="{ACCENT}">Source: {label}</a>'

        s = re.sub(r"\[\[Source:\s*([^\]]+)\]\]\((https?://[^)]+)\)", link_repl, s)

        # Convert **bold** -> <b>bold</b>
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

        return s

    def _fmt_money(x) -> str:
        try:
            return f"${float(x):,.2f}"
        except Exception:
            return "n/a"

    def _fmt_int(x) -> str:
        try:
            return f"{int(float(x)):,}"
        except Exception:
            return "n/a"

    def _fmt_num(x) -> str:
        try:
            return f"{float(x):,.2f}"
        except Exception:
            return "n/a"

    # -----------------------------
    # 4) Document + typography
    # -----------------------------
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.80 * inch,
        bottomMargin=0.80 * inch,
        title=f"{ticker} Due Diligence",
        author="Stock Research Bot",
    )

    base = getSampleStyleSheet()

    # Title styles
    S_TITLE = ParagraphStyle(
        "S_TITLE",
        parent=base["Title"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=10,
    )

    S_SUBTITLE = ParagraphStyle(
        "S_SUBTITLE",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=11.5,
        leading=16,
        textColor=MUTED,
        alignment=TA_LEFT,
        spaceAfter=14,
    )

    # Section headers (clean, modern spacing)
    S_H1 = ParagraphStyle(
        "S_H1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=ACCENT_DARK,
        spaceBefore=14,
        spaceAfter=8,
    )

    S_H2 = ParagraphStyle(
        "S_H2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12.5,
        leading=16,
        textColor=INK,
        spaceBefore=12,
        spaceAfter=6,
    )

    # Body
    S_BODY = ParagraphStyle(
        "S_BODY",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10.6,
        leading=15,
        textColor=INK,
        spaceAfter=6,
    )

    S_BODY_MUTED = ParagraphStyle(
        "S_BODY_MUTED",
        parent=S_BODY,
        textColor=MUTED,
        fontSize=9.8,
        leading=13.5,
    )

    S_KPI = ParagraphStyle(
        "S_KPI",
        parent=base["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=INK,
        alignment=TA_LEFT,
    )

    S_KPI_LABEL = ParagraphStyle(
        "S_KPI_LABEL",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=11,
        textColor=MUTED,
        alignment=TA_LEFT,
    )

    S_FOOTER = ParagraphStyle(
        "S_FOOTER",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=10,
        textColor=MUTED,
        alignment=TA_RIGHT,
    )

    # Bullets / numbered items
    S_LIST = ParagraphStyle(
        "S_LIST",
        parent=S_BODY,
        leftIndent=0,
        spaceAfter=2,
    )

    # -----------------------------
    # 5) Header / footer renderer
    # -----------------------------
    def _draw_header_footer(canvas, doc_obj):
        """
        A small bit of polish: consistent header + page numbers.
        (This is the part people subconsciously read as “professional”.)
        """
        canvas.saveState()

        page_w, page_h = LETTER

        # Header line
        y = page_h - 0.55 * inch
        canvas.setStrokeColor(SOFT)
        canvas.setLineWidth(1)
        canvas.line(doc.leftMargin, y, page_w - doc.rightMargin, y)

        # Header text (left)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(doc.leftMargin, y + 8, f"{ticker} • Due Diligence")

        # Header text (right)
        right_text = generated_human
        tw = stringWidth(right_text, "Helvetica", 9)
        canvas.drawString(page_w - doc.rightMargin - tw, y + 8, right_text)

        # Footer: page number
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 8.5)
        canvas.drawRightString(page_w - doc.rightMargin, 0.55 * inch, f"Page {doc_obj.page}")

        canvas.restoreState()

    # -----------------------------
    # 6) Cover page "cards"
    # -----------------------------
    rating = report.get("rating", "n/a")
    confidence = report.get("confidence", "n/a")
    try:
        conf_pct = f"{float(confidence) * 100:.0f}%"
    except Exception:
        conf_pct = str(confidence)

    # Deterministic snapshot (compact facts)
    det = out.get("deterministic") or {}
    det_meta = out.get("deterministic_meta") or {}
    snap = det.get("snapshot") or det.get("market_snapshot") or det.get("compact_snapshot") or {}

    company_name = snap.get("company_name") or det.get("company_name") or ""
    sector = snap.get("sector") or det.get("sector") or ""
    industry = snap.get("industry") or det.get("industry") or ""

    price = snap.get("price") or out.get("current_price")
    prev_close = snap.get("previous_close") or snap.get("prev_close")
    volume = snap.get("volume")
    as_of = snap.get("as_of") or ""

    shares_out = snap.get("shares_outstanding")
    mcap_rep = snap.get("market_cap_reported")
    mcap_calc = snap.get("market_cap_calculated")
    net_debt = snap.get("net_debt")
    fcf_ttm = snap.get("free_cash_flow_ttm") or snap.get("fcf_ttm")

    # Build a “KPI row” as a table (cleaner than bullets for top stats)
    def kpi_cell(label: str, value: str) -> List:
        return [
            Paragraph(_escape_rl(value), S_KPI),
            Spacer(1, 2),
            Paragraph(_escape_rl(label), S_KPI_LABEL),
        ]

    kpi_table = Table(
        [
            [
                kpi_cell("Rating", str(rating)),
                kpi_cell("Confidence", conf_pct),
                kpi_cell("Price", _fmt_money(price)),
                kpi_cell("Volume", _fmt_int(volume) if volume is not None else "n/a"),
            ]
        ],
        colWidths=[(doc.width / 4.0)] * 4,
    )
    kpi_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("BACKGROUND", (0, 0), (-1, -1), PANEL_BG),
                ("BOX", (0, 0), (-1, -1), 1, SOFT),
                ("LINEBELOW", (0, 0), (-1, 0), 0, SOFT),
            ]
        )
    )

    # A compact “facts card” under the KPI row
    facts_rows = []

    # A little human touch: only show rows that are present.
    def add_fact(label: str, value: str):
        if value and value.strip() and value.strip().lower() not in {"none", "n/a", "nan", "null"}:
            facts_rows.append([Paragraph(f"<b>{_escape_rl(label)}</b>", S_BODY_MUTED), Paragraph(_escape_rl(value), S_BODY)])

    add_fact("Company", company_name.strip() or "n/a")
    if sector or industry:
        add_fact("Sector / Industry", f"{sector or 'n/a'} / {industry or 'n/a'}")
    add_fact("As of", str(as_of) if as_of else "n/a")
    if prev_close is not None:
        add_fact("Prev Close", _fmt_money(prev_close))
    if shares_out is not None:
        add_fact("Shares Outstanding", _fmt_int(shares_out))
    if mcap_rep is not None:
        add_fact("Market Cap (reported)", _fmt_money(mcap_rep))
    if mcap_calc is not None:
        add_fact("Market Cap (calculated)", _fmt_money(mcap_calc))
    if net_debt is not None:
        add_fact("Net Debt", _fmt_money(net_debt))
    if fcf_ttm is not None:
        add_fact("FCF (TTM)", _fmt_money(fcf_ttm))

    facts_table = Table(
        facts_rows if facts_rows else [[Paragraph("<b>Snapshot</b>", S_BODY_MUTED), Paragraph("No deterministic facts available.", S_BODY)]],
        colWidths=[doc.width * 0.28, doc.width * 0.72],
    )
    facts_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#FFFFFF")),
                ("BOX", (0, 0), (-1, -1), 1, SOFT),
                ("LINEBELOW", (0, 0), (-1, -2), 0.5, SOFT),
            ]
        )
    )

    # -----------------------------
    # 7) Turn markdown into Flowables
    # -----------------------------
    elements: List[Any] = []

    # Cover (page 1)
    title_line = f"{ticker} Due Diligence"
    subtitle_line = "Evidence-backed summary built from deterministic market data + retrieved sources."

    elements.append(Paragraph(_escape_rl(title_line), S_TITLE))
    elements.append(Paragraph(_escape_rl(subtitle_line), S_SUBTITLE))
    elements.append(Spacer(1, 6))
    elements.append(kpi_table)
    elements.append(Spacer(1, 10))
    elements.append(KeepTogether([Paragraph("Deterministic Snapshot", S_H2), facts_table]))

    # If there are deterministic warnings, surface them *politely* on the cover
    warnings_list = (det_meta.get("warnings") or []) if isinstance(det_meta, dict) else []
    if warnings_list:
        preview = warnings_list[:4]
        warn_items = [ListItem(Paragraph(_md_inline_to_rl(str(w)), S_BODY), leftIndent=14) for w in preview]
        warn_block = KeepTogether(
            [
                Spacer(1, 10),
                Paragraph("Data Warnings (preview)", S_H2),
                ListFlowable(warn_items, bulletType="bullet", leftIndent=14),
                Paragraph(_escape_rl(f"({len(warnings_list)} total warnings captured during data collection.)"), S_BODY_MUTED),
            ]
        )
        elements.append(warn_block)

    elements.append(PageBreak())

    # Body (pages 2+)
    # We'll do a tiny markdown parser: headings, bullets, numbered, HR, paragraphs.
    bullet_buf: List[ListItem] = []
    number_buf: List[ListItem] = []

    def flush_lists():
        nonlocal bullet_buf, number_buf
        if bullet_buf:
            elements.append(ListFlowable(bullet_buf, bulletType="bullet", leftIndent=16))
            bullet_buf = []
            elements.append(Spacer(1, 6))
        if number_buf:
            elements.append(ListFlowable(number_buf, bulletType="1", leftIndent=18))
            number_buf = []
            elements.append(Spacer(1, 6))

    # Sometimes the model outputs short “label:” lines; treat as paragraph, not a header.
    def is_hr(line: str) -> bool:
        s = line.strip()
        return s in {"---", "***"} or (len(s) >= 3 and set(s) == {"-"}) or (len(s) >= 3 and set(s) == {"*"})

    for raw in md.splitlines():
        line = (raw or "").rstrip("\n")
        s = line.strip()

        # Blank line => spacing break, and a natural place to flush lists
        if not s:
            flush_lists()
            elements.append(Spacer(1, 8))
            continue

        # Horizontal rule
        if is_hr(s):
            flush_lists()
            elements.append(Spacer(1, 6))
            elements.append(HRFlowable(width="100%", thickness=1, color=SOFT, spaceBefore=6, spaceAfter=10))
            continue

        # Headings
        if s.startswith("### "):
            flush_lists()
            elements.append(Paragraph(_md_inline_to_rl(s[4:]), S_H2))
            continue

        if s.startswith("## "):
            flush_lists()
            elements.append(Paragraph(_md_inline_to_rl(s[3:]), S_H1))
            continue

        if s.startswith("# "):
            # If a stray H1 appears, treat it like your main section header.
            flush_lists()
            elements.append(Paragraph(_md_inline_to_rl(s[2:]), S_H1))
            continue

        # Bullets: "- "
        if s.startswith("- "):
            txt = _md_inline_to_rl(s[2:].strip())
            bullet_buf.append(ListItem(Paragraph(txt, S_LIST), leftIndent=16))
            continue

        # Numbered list: "1. "
        m_num = re.match(r"^(\d+)\.\s+(.*)$", s)
        if m_num:
            txt = _md_inline_to_rl(m_num.group(2).strip())
            number_buf.append(ListItem(Paragraph(txt, S_LIST), leftIndent=16))
            continue

        # Normal paragraph
        flush_lists()
        elements.append(Paragraph(_md_inline_to_rl(s), S_BODY))

    flush_lists()

    # -----------------------------
    # 8) Build PDF
    # -----------------------------
    # ReportLab lets us reuse the same draw function for all pages.
    doc.build(
        elements,
        onFirstPage=_draw_header_footer,
        onLaterPages=_draw_header_footer,
    )

    return pdf_path
