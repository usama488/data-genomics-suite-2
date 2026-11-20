"""Radiology-style PDF report for an AI-assisted chest X-ray read."""
from __future__ import annotations

import datetime as dt
import io

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SS = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=SS["Heading1"], fontSize=16, textColor=colors.HexColor("#0B5345"), spaceAfter=3)
H2 = ParagraphStyle("h2", parent=SS["Heading2"], fontSize=12, textColor=colors.HexColor("#0B5345"), spaceBefore=8, spaceAfter=4)
BODY = ParagraphStyle("b", parent=SS["BodyText"], fontSize=9, leading=12)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#555555"))
CELL = ParagraphStyle("c", parent=BODY, fontSize=8, leading=10)
CELLH = ParagraphStyle("ch", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")
BAN = ParagraphStyle("ban", parent=BODY, fontSize=11.5, leading=15, alignment=TA_CENTER, textColor=colors.white)
TRIAGE_COL = {"STAT": "#C0392B", "Urgent": "#CA6F1E", "Routine": "#2874A6", "Normal": "#1E8449"}


def _tab(rows, widths):
    rows = [[Paragraph(str(h), CELLH) for h in rows[0]]] + [[Paragraph(str(c), CELL) for c in r] for r in rows[1:]]
    t = Table(rows, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B5345")), ("GRID", (0, 0), (-1, -1), .25, colors.HexColor("#BBBBBB")),
          ("VALIGN", (0, 0), (-1, -1), "TOP")]
    for i in range(2, len(rows), 2):
        st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#E8F6F3")))
    t.setStyle(TableStyle(st))
    return t


def build(study: dict, lab: dict, findings, decisions: dict, impression: str, triage: str, images: list[tuple[bytes, str]],
          ctr: float | None, qc_flags: list[dict]) -> bytes:
    p = study.get("patient", {})
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=13 * mm, bottomMargin=15 * mm,
                            title=f"Chest X-ray report {p.get('id', '')}")

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.HexColor("#777777"))
        c.drawString(14 * mm, 8 * mm, f"{lab.get('name', '')} · {p.get('id', '')} · AI-assisted report · {dt.date.today()}")
        c.drawRightString(A4[0] - 14 * mm, 8 * mm, f"Page {d.page}")
        c.restoreState()

    el = [Paragraph(lab.get("name") or "Department of Radiology", H1), Paragraph(lab.get("address", ""), SMALL),
          Paragraph("<b>Chest radiograph - AI-assisted report</b>", BODY), Spacer(1, 4)]
    info = [["Patient", p.get("name", ""), "Patient ID / MRN", p.get("id", "")],
            ["Age / Sex", f"{p.get('age', '')} / {p.get('sex', '')}", "Study date", p.get("date", str(dt.date.today()))],
            ["Examination", p.get("exam", "Chest X-ray"), "View", p.get("view", "")],
            ["Clinical indication", Paragraph(p.get("indication", ""), CELL), "Referring clinician", p.get("clinician", "")]]
    t = Table(info, colWidths=[32 * mm, 60 * mm, 34 * mm, 56 * mm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F6F3")),
                           ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#E8F6F3")), ("GRID", (0, 0), (-1, -1), .25, colors.grey),
                           ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    el += [t, Spacer(1, 7)]
    confirmed = [f for f, d in decisions.items() if d == "Confirmed"]
    b = Table([[Paragraph(f"<b>AI TRIAGE: {triage.upper()}</b> &nbsp;·&nbsp; Radiologist-confirmed findings: "
                          f"{', '.join(confirmed) if confirmed else 'none'}", BAN)]], colWidths=[182 * mm])
    b.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(TRIAGE_COL.get(triage, "#555"))),
                           ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    el.append(b)

    el.append(Paragraph("Impression", H2))
    el.append(Paragraph((impression or "No impression entered.").replace("\n", "<br/>"), BODY))
    if images:
        el.append(Paragraph("Images", H2))
        cells = []
        for data, cap in images[:2]:
            im = RLImage(io.BytesIO(data), width=86 * mm, height=86 * mm)
            cells.append([im, Paragraph(cap, SMALL)])
        row = [Table([[c[0]], [c[1]]], colWidths=[88 * mm]) for c in cells]
        el.append(Table([row], colWidths=[90 * mm] * len(row)))
    el.append(Paragraph("AI findings and radiologist review", H2))
    rows = [["Finding", "AI score", "AI call", "Radiologist", "Meaning"]]
    for r in findings.itertuples():
        if r.score >= .4 or r.finding in decisions:
            rows.append([r.finding.replace("_", " "), f"{r.score:.2f}", str(r.confidence), decisions.get(r.finding, "not reviewed"), r.description])
    el.append(_tab(rows, [36 * mm, 18 * mm, 22 * mm, 26 * mm, 80 * mm]))
    el.append(Paragraph("AI score: 0.5 = the model's operating threshold; scores further above 0.5 indicate higher confidence. "
                        "Findings below 0.4 are omitted.", SMALL))
    if ctr:
        el.append(Paragraph("Measurements", H2))
        el.append(Paragraph(f"Cardiothoracic ratio (manual): <b>{ctr:.2f}</b> - "
                            + ("above 0.50, consistent with cardiomegaly on a PA film." if ctr > .5 else "within normal limits (≤ 0.50)."), BODY))
    el.append(Paragraph("Image quality", H2))
    el.append(Paragraph("; ".join(f["text"] for f in qc_flags), BODY))
    el.append(Paragraph("About the AI model", H2))
    el.append(Paragraph("DenseNet-121 convolutional neural network from TorchXRayVision (Cohen et al., MIDL 2022) trained on eight public "
                        "chest X-ray datasets (NIH ChestX-ray14, CheXpert, MIMIC-CXR, PadChest, OpenI, Google, RSNA). It analyses frontal "
                        "(PA/AP) adult chest radiographs at 224×224 pixels and outputs 18 findings. Heat-maps are class-activation maps and "
                        "show regions that influenced the score, not lesion boundaries.", BODY))
    el.append(Paragraph("<b>Important:</b> research / education software - not a cleared medical device. The AI output is a decision aid "
                        "only; the final report is the responsibility of the reporting radiologist.", SMALL))
    el.append(Spacer(1, 12))
    sig = Table([["Reported by:", lab.get("radiologist", ""), "Signature:", ""], ["Date / time:", dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "", ""]],
                colWidths=[28 * mm, 70 * mm, 24 * mm, 60 * mm])
    sig.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("LINEBELOW", (3, 0), (3, 0), .5, colors.black)]))
    el.append(KeepTogether(sig))
    doc.build(el, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
