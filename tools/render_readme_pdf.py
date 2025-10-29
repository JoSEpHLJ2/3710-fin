"""
Render a Markdown README to PDF using ReportLab.

This lightweight renderer supports headings, bullet lists, tables,
inline paragraphs, fenced code blocks (monospace text), and images
referenced with standard Markdown syntax.

Usage:
    python tools/render_readme_pdf.py recognition/vqvae2_hipmri/README.md

The PDF will be written to the same directory as the source Markdown
with the same basename (`README.pdf`).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.pdfgen import canvas


def render_markdown(md_path: Path, pdf_path: Path) -> None:
    lines = md_path.read_text(encoding="utf-8").splitlines()
    canv = canvas.Canvas(str(pdf_path), pagesize=A4)
    page_width, page_height = A4

    margin = 40
    x_origin = margin
    current_y = page_height - margin

    normal_font = ("Helvetica", 10)
    bold_font = ("Helvetica-Bold", 12)
    mono_font = ("Courier", 9)

    def reset_page() -> None:
        nonlocal current_y
        current_y = page_height - margin

    def ensure_space(leading: float) -> None:
        nonlocal current_y
        if current_y < margin + leading:
            canv.showPage()
            reset_page()

    def draw_text(text_line: str, leading: float, font: tuple[str, float]) -> None:
        nonlocal current_y
        ensure_space(leading)
        canv.setFont(*font)
        canv.drawString(x_origin, current_y, text_line)
        current_y -= leading

    def draw_wrapped(paragraph: str, font: tuple[str, float]) -> None:
        nonlocal current_y
        wrapped = simpleSplit(paragraph, font[0], font[1], page_width - 2 * margin)
        for part in wrapped:
            draw_text(part, font[1] + 4, font)

    def draw_heading(text_line: str, level: int) -> None:
        size = 18 if level == 1 else 14 if level == 2 else 12
        draw_text(text_line, size + 6, ("Helvetica-Bold", size))

    def draw_bullet(text_line: str, indent_level: int) -> None:
        nonlocal current_y
        leading = normal_font[1] + 4
        ensure_space(leading)
        bullet_x = x_origin + indent_level * 15
        canv.setFont(*normal_font)
        canv.drawString(bullet_x, current_y, "• " + text_line)
        current_y -= leading

    def draw_code_block(lines_: list[str]) -> None:
        canv.setFont(*mono_font)
        for code_line in lines_:
            draw_text(code_line, mono_font[1] + 4, mono_font)

    def draw_table(table_lines: list[str]) -> None:
        nonlocal current_y
        headers = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
        rows = [[cell.strip() for cell in row.strip("|").split("|")] for row in table_lines[2:]]
        col_count = len(headers)
        col_width = (page_width - 2 * margin) / max(1, col_count)
        row_height = normal_font[1] + 6
        required = (len(rows) + 1) * row_height + 12
        if current_y < margin + required:
            canv.showPage()
            reset_page()
        canv.setFont(*bold_font)
        for idx, header in enumerate(headers):
            canv.drawString(x_origin + idx * col_width, current_y, header)
        current_y -= row_height
        canv.setStrokeColor(colors.grey)
        canv.line(x_origin, current_y + row_height / 2, x_origin + col_count * col_width, current_y + row_height / 2)
        canv.setStrokeColor(colors.black)
        canv.setFont(*normal_font)
        for row in rows:
            for idx, cell in enumerate(row):
                canv.drawString(x_origin + idx * col_width, current_y, cell)
            current_y -= row_height
        current_y -= 6

    def draw_image(path: str, caption: str | None) -> None:
        nonlocal current_y
        img_path = md_path.parent / path
        if not img_path.exists():
            return
        try:
            img = ImageReader(str(img_path))
        except Exception:
            return
        iw, ih = img.getSize()
        max_width = page_width - 2 * margin
        scale = min(1.0, max_width / iw)
        render_w = iw * scale
        render_h = ih * scale
        if current_y < margin + render_h + 40:
            canv.showPage()
            reset_page()
        y_bottom = current_y - render_h
        canv.drawImage(img, x_origin, y_bottom, width=render_w, height=render_h, preserveAspectRatio=True, mask="auto")
        current_y = y_bottom - 6
        if caption:
            canv.setFont("Helvetica-Oblique", 9)
            for part in simpleSplit(caption, "Helvetica-Oblique", 9, max_width):
                ensure_space(12)
                canv.drawString(x_origin, current_y, part)
                current_y -= 12

    para_buffer: list[str] = []
    code_buffer: list[str] = []
    i = 0
    in_code = False
    while i < len(lines):
        raw = lines[i]
        image_match = re.match(r"!\\[(.*?)\\]\\((.*?)\\)", raw)
        heading_match = re.match(r"^(#{1,6})\\s+(.*)$", raw)
        bullet_match = re.match(r"^\\s*[-*+]\\s+(.*)$", raw)

        if raw.strip().startswith("```"):
            if in_code:
                draw_code_block(code_buffer)
                code_buffer.clear()
                in_code = False
            else:
                if para_buffer:
                    draw_wrapped(" ".join(para_buffer), normal_font)
                    para_buffer.clear()
                in_code = True
            i += 1
            continue

        if in_code:
            code_buffer.append(raw)
            i += 1
            continue

        if image_match:
            if para_buffer:
                draw_wrapped(" ".join(para_buffer), normal_font)
                para_buffer.clear()
            alt, path = image_match.groups()
            draw_image(path, alt or None)
            i += 1
            continue

        if heading_match:
            if para_buffer:
                draw_wrapped(" ".join(para_buffer), normal_font)
                para_buffer.clear()
            draw_heading(heading_match.group(2), len(heading_match.group(1)))
            i += 1
            continue

        if bullet_match:
            if para_buffer:
                draw_wrapped(" ".join(para_buffer), normal_font)
                para_buffer.clear()
            indent = (len(raw) - len(raw.lstrip())) // 2
            draw_bullet(bullet_match.group(1), indent)
            i += 1
            continue

        if raw.startswith("|") and i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"|", "-", " "}:
            table_lines = [raw]
            j = i + 1
            while j < len(lines) and lines[j].startswith("|"):
                table_lines.append(lines[j])
                j += 1
            draw_table(table_lines)
            i = j
            continue

        if raw.strip() == "---":
            if para_buffer:
                draw_wrapped(" ".join(para_buffer), normal_font)
                para_buffer.clear()
            ensure_space(20)
            canv.setStrokeColor(colors.grey)
            canv.line(margin, current_y, page_width - margin, current_y)
            canv.setStrokeColor(colors.black)
            current_y -= 12
            i += 1
            continue

        if not raw.strip():
            if para_buffer:
                draw_wrapped(" ".join(para_buffer), normal_font)
                para_buffer.clear()
            current_y -= normal_font[1]
            i += 1
            continue

        para_buffer.append(raw)
        i += 1

    if para_buffer:
        draw_wrapped(" ".join(para_buffer), normal_font)
    if code_buffer:
        draw_code_block(code_buffer)

    canv.save()


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    md_path = Path(sys.argv[1])
    if not md_path.exists():
        raise FileNotFoundError(md_path)
    pdf_path = md_path.with_suffix(".pdf")
    render_markdown(md_path, pdf_path)
    print(f"Saved PDF to {pdf_path}")


if __name__ == "__main__":
    main()

