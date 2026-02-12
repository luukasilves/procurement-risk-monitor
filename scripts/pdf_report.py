"""
PDF Report Generator for Procurement Risk Analysis
====================================================
Generates professional PDF reports for individual procurement deep dives.
This is the €20-30 product — the analysis that saves a procurement officer
hours of manual review and helps prevent costly VAKO disputes.

Uses fpdf2 (pure Python, no system dependencies).
"""
from __future__ import annotations

import io
import math
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

# Brand colors
NAVY = (15, 30, 61)
BLUE = (37, 99, 235)
LIGHT_BLUE = (219, 234, 254)
GREEN = (22, 163, 74)
LIGHT_GREEN = (220, 252, 231)
AMBER = (245, 158, 11)
LIGHT_AMBER = (254, 243, 199)
RED = (220, 38, 38)
LIGHT_RED = (254, 226, 226)
GRAY = (107, 114, 128)
LIGHT_GRAY = (243, 244, 246)
WHITE = (255, 255, 255)
DARK = (31, 41, 55)


def _severity_color(severity: str):
    if severity in ("high", "HIGH"):
        return RED, LIGHT_RED
    if severity in ("medium", "MEDIUM"):
        return AMBER, LIGHT_AMBER
    return GREEN, LIGHT_GREEN


class ProcurementReport(FPDF):
    """Professional procurement risk analysis PDF report."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=25)

    def header(self):
        if self.page_no() == 1:
            return  # Cover page has custom header
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*GRAY)
        self.cell(0, 8, "ProcureSight  |  Procurement Risk Analysis", align="L")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 8, f"Page {self.page_no()}", align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-20)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 5,
                  f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  "
                  "procuresight.ee  |  Confidential",
                  align="C")

    # ── Cover Page ──────────────────────────────────────────────

    def cover_page(self, title: str, buyer: str, rhr_id: str,
                   risk_score: float | None, risk_label_text: str,
                   value_str: str, procedure: str, sector: str):
        self.add_page()
        # Top bar
        self.set_fill_color(*NAVY)
        self.rect(0, 0, 210, 60, "F")

        # Brand
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*WHITE)
        self.set_xy(15, 12)
        self.cell(0, 12, "PROCURESIGHT", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 11)
        self.set_xy(15, 28)
        self.cell(0, 8, "Procurement Risk Intelligence", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_xy(15, 38)
        self.cell(0, 6, f"Analysis Report  |  {datetime.now().strftime('%d %B %Y')}")

        # Title section
        self.set_text_color(*DARK)
        self.set_xy(15, 75)
        self.set_font("Helvetica", "B", 18)
        # Truncate long titles
        display_title = title[:100] + "..." if len(title) > 100 else title
        self.multi_cell(180, 9, display_title)

        self.ln(4)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(*GRAY)
        info_line = f"{buyer}  |  {procedure}  |  {value_str}  |  {sector}"
        self.multi_cell(180, 7, info_line)
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, f"RHR ID: {rhr_id}", new_x="LMARGIN", new_y="NEXT")

        # Risk score box
        self.ln(10)
        if risk_score is not None:
            pct = risk_score * 100
            if pct >= 15:
                bg, fg = LIGHT_RED, RED
            elif pct >= 8:
                bg, fg = LIGHT_AMBER, AMBER
            elif pct >= 4:
                bg, fg = LIGHT_AMBER, AMBER
            else:
                bg, fg = LIGHT_GREEN, GREEN

            box_y = self.get_y()
            self.set_fill_color(*bg)
            self.rect(15, box_y, 180, 35, "F")

            self.set_xy(25, box_y + 3)
            self.set_font("Helvetica", "", 10)
            self.set_text_color(*GRAY)
            self.cell(60, 7, "RISK ASSESSMENT")

            self.set_xy(25, box_y + 12)
            self.set_font("Helvetica", "B", 24)
            self.set_text_color(*fg)
            self.cell(60, 12, f"{pct:.1f}%")

            self.set_xy(70, box_y + 14)
            self.set_font("Helvetica", "B", 14)
            self.cell(60, 10, risk_label_text.upper())

            # Context text
            self.set_xy(120, box_y + 8)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*DARK)
            self.multi_cell(70, 5,
                            "Based on 38-feature ML model trained on 57,000+ "
                            "procurements and 959 VAKO disputes (2018-2026).")

            self.set_y(box_y + 40)

        # Table of contents
        self.ln(8)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*DARK)
        self.cell(0, 8, "Contents", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.line(15, self.get_y(), 60, self.get_y())
        self.ln(4)

        toc_items = [
            "1. Executive Summary",
            "2. Procurement Quality Assessment",
            "3. Risk Factor Analysis",
            "4. Integrity Checks",
            "5. Legal Compliance Review",
            "6. Buyer Profile & Benchmarking",
            "7. Comparable Procurements",
            "8. Recommendations",
        ]
        self.set_font("Helvetica", "", 10)
        for item in toc_items:
            self.cell(0, 6, item, new_x="LMARGIN", new_y="NEXT")

    # ── Section helpers ─────────────────────────────────────────

    def section_title(self, number: int, title: str):
        self.ln(6)
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*NAVY)
        self.cell(0, 10, f"{number}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.7)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)
        self.set_text_color(*DARK)
        self.set_font("Helvetica", "", 10)

    def subsection(self, title: str):
        self.ln(3)
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*BLUE)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(*DARK)
        self.set_font("Helvetica", "", 10)
        self.ln(1)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def key_value_row(self, key: str, value: str, bold_value: bool = False):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*GRAY)
        self.cell(55, 6, key)
        self.set_text_color(*DARK)
        self.set_font("Helvetica", "B" if bold_value else "", 10)
        self.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")

    def info_box(self, text: str, severity: str = "info"):
        fg, bg = _severity_color(severity)
        if severity == "info":
            bg = LIGHT_BLUE
            fg = BLUE
        self.set_fill_color(*bg)
        y = self.get_y()
        self.rect(15, y, 180, max(12, len(text) / 8), "F")
        self.set_xy(18, y + 2)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*fg)
        self.multi_cell(174, 4.5, text)
        self.set_y(max(self.get_y() + 2, y + 14))
        self.set_text_color(*DARK)

    def recommendation_box(self, priority: str, action: str, rationale: str):
        fg, bg = _severity_color(priority)
        y = self.get_y()
        if y > 260:
            self.add_page()
            y = self.get_y()
        # Priority badge
        self.set_fill_color(*fg)
        self.set_xy(15, y)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*WHITE)
        badge_w = self.get_string_width(priority) + 6
        self.cell(badge_w, 6, priority, fill=True)

        # Action title
        self.set_xy(15 + badge_w + 3, y)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*DARK)
        self.cell(0, 6, action, new_x="LMARGIN", new_y="NEXT")

        # Rationale
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*GRAY)
        self.multi_cell(175, 4.5, rationale, new_x="LMARGIN")
        self.ln(3)

    def simple_table(self, headers: list[str], rows: list[list[str]],
                     col_widths: list[int] | None = None):
        if not col_widths:
            total_w = 180
            col_widths = [total_w // len(headers)] * len(headers)
            # Give last column remaining space
            col_widths[-1] = total_w - sum(col_widths[:-1])

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=0, fill=True)
        self.ln()

        # Rows
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*DARK)
        for ri, row in enumerate(rows):
            if ri % 2 == 0:
                self.set_fill_color(*LIGHT_GRAY)
            else:
                self.set_fill_color(*WHITE)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell)[:40], border=0, fill=True)
            self.ln()
        self.ln(3)


def generate_report(
    rhr_id: str,
    title: str,
    buyer: str,
    risk_score: float | None,
    risk_label_text: str,
    value_str: str,
    procedure: str,
    procedure_label: str,
    contract_type: str,
    sector: str,
    sector_label: str,
    features: dict,
    feature_names: list[str],
    contributions: list[tuple[str, str, float]],  # (name, label, contribution)
    integrity_flags: list[tuple[str, str, str]],  # (name, description, severity)
    checklist: list[tuple[str, str, str]],  # (priority, action, rationale)
    buyer_profile: dict | None,
    sector_benchmarks: dict | None,
    comparable_procs: list[dict],
    dispute_details: list[dict],
    quality_assessment: dict,  # from compute_quality_assessment
) -> bytes:
    """Generate a complete PDF report and return as bytes."""
    pdf = ProcurementReport()

    # ── Page 1: Cover ──────────────────────────────────────────
    pdf.cover_page(
        title=title or f"Procurement {rhr_id}",
        buyer=buyer or "Unknown",
        rhr_id=rhr_id,
        risk_score=risk_score,
        risk_label_text=risk_label_text,
        value_str=value_str,
        procedure=procedure_label,
        sector=sector_label,
    )

    # ── Section 1: Executive Summary ───────────────────────────
    pdf.add_page()
    pdf.section_title(1, "Executive Summary")

    # Build summary text
    summary_parts = []
    if risk_score is not None:
        pct = risk_score * 100
        summary_parts.append(
            f"This procurement carries a {pct:.1f}% dispute risk score "
            f"({risk_label_text.lower()} risk), placing it in the "
            f"{'top 5%' if pct > 85 else 'top 15%' if pct > 70 else 'middle range'} "
            f"of all Estonian procurements."
        )
    if integrity_flags:
        n_high = sum(1 for _, _, s in integrity_flags if s in ("high", "HIGH"))
        if n_high:
            summary_parts.append(
                f"Integrity analysis identified {n_high} high-priority flag(s) "
                "that warrant immediate attention."
            )
    if quality_assessment:
        qs = quality_assessment.get("overall_score", 0)
        summary_parts.append(
            f"The overall procurement quality score is {qs}/100. "
            + quality_assessment.get("summary", "")
        )
    if dispute_details:
        summary_parts.append(
            f"This procurement has {len(dispute_details)} VAKO dispute(s) on record."
        )

    for part in summary_parts:
        pdf.body_text(part)

    # Key facts table
    pdf.subsection("Key Facts")
    pdf.key_value_row("Contracting Authority", buyer or "Not specified")
    pdf.key_value_row("Contract Type", contract_type.title() if contract_type else "Not specified")
    pdf.key_value_row("Procedure", procedure_label)
    pdf.key_value_row("Sector", sector_label)
    pdf.key_value_row("Estimated Value", value_str, bold_value=True)
    pw = features.get("price_weight", 0)
    qw = features.get("quality_weight", 0)
    if pw or qw:
        total_w = pw + qw
        if total_w > 0:
            pdf.key_value_row("Price / Quality Split",
                              f"{pw / total_w * 100:.0f}% / {qw / total_w * 100:.0f}%")
    flags_list = []
    if features.get("is_eu_funded"):
        flags_list.append("EU Funded")
    if features.get("is_framework"):
        flags_list.append("Framework Agreement")
    if features.get("has_green"):
        flags_list.append("Green Criteria")
    if features.get("has_innovation"):
        flags_list.append("Innovation Criteria")
    if features.get("has_social"):
        flags_list.append("Social Criteria")
    if flags_list:
        pdf.key_value_row("Flags", ", ".join(flags_list))

    # ── Section 2: Procurement Quality Assessment ──────────────
    pdf.add_page()
    pdf.section_title(2, "Procurement Quality Assessment")

    if quality_assessment:
        qa = quality_assessment
        pdf.body_text(
            "This assessment evaluates the procurement against best practices "
            "in five dimensions. Each dimension is scored 0-20, totalling 0-100."
        )

        # Dimension scores table
        dimensions = qa.get("dimensions", {})
        if dimensions:
            headers = ["Dimension", "Score", "Rating", "Key Finding"]
            rows = []
            for dim_name, dim_data in dimensions.items():
                score = dim_data.get("score", 0)
                max_score = dim_data.get("max", 20)
                rating = "Excellent" if score >= 16 else "Good" if score >= 12 else "Fair" if score >= 8 else "Needs Work"
                finding = dim_data.get("finding", "")[:40]
                rows.append([dim_name, f"{score}/{max_score}", rating, finding])
            pdf.simple_table(headers, rows, [45, 20, 25, 90])

        overall = qa.get("overall_score", 0)
        pdf.ln(3)
        if overall >= 80:
            pdf.info_box(f"Overall Quality: {overall}/100 - Excellent. This procurement demonstrates best practices.", "info")
        elif overall >= 60:
            pdf.info_box(f"Overall Quality: {overall}/100 - Good with room for improvement.", "info")
        elif overall >= 40:
            pdf.info_box(f"Overall Quality: {overall}/100 - Several areas need attention before publication.", "medium")
        else:
            pdf.info_box(f"Overall Quality: {overall}/100 - Significant improvements recommended.", "high")

        # Detailed dimension write-ups
        for dim_name, dim_data in dimensions.items():
            pdf.subsection(dim_name)
            pdf.body_text(dim_data.get("explanation", ""))
    else:
        pdf.body_text("Quality assessment not available for this procurement.")

    # ── Section 3: Risk Factor Analysis ────────────────────────
    pdf.add_page()
    pdf.section_title(3, "Risk Factor Analysis")
    pdf.body_text(
        "The ML model identifies which features of this procurement increase or "
        "decrease its dispute probability relative to the average procurement."
    )

    if contributions:
        # Top risk-increasing
        increasing = [(n, l, c) for n, l, c in contributions if c > 0.02]
        decreasing = [(n, l, c) for n, l, c in contributions if c < -0.02]

        if increasing:
            pdf.subsection("Risk-Increasing Factors")
            headers = ["Factor", "Impact", "Direction"]
            rows = []
            for name, label, contrib in sorted(increasing, key=lambda x: -x[2])[:8]:
                rows.append([label, f"{contrib:+.3f}", "Increases risk"])
            pdf.simple_table(headers, rows, [80, 30, 70])

        if decreasing:
            pdf.subsection("Risk-Decreasing Factors")
            headers = ["Factor", "Impact", "Direction"]
            rows = []
            for name, label, contrib in sorted(decreasing, key=lambda x: x[2])[:8]:
                rows.append([label, f"{contrib:+.3f}", "Decreases risk"])
            pdf.simple_table(headers, rows, [80, 30, 70])

    # ── Section 4: Integrity Checks ────────────────────────────
    pdf.section_title(4, "Integrity Checks")
    pdf.body_text(
        "Cross-referencing the winning company and procurement structure against "
        "external registries: political donor records (ERJK), company registry "
        "(e-Ariregister), and beneficial ownership data."
    )

    if integrity_flags:
        for flag_name, flag_desc, severity in integrity_flags:
            pdf.recommendation_box(severity.upper(), flag_name, flag_desc)
    else:
        pdf.info_box(
            "No integrity flags identified. The winning company has no material "
            "political donor links, no hidden ownership concentration, and normal "
            "price-to-category ratios.",
            "info"
        )

    # ── Section 5: Legal Compliance Review ─────────────────────
    pdf.add_page()
    pdf.section_title(5, "Legal Compliance Review")

    compliance_items = _build_compliance_review(features, procedure, contract_type, sector)
    if compliance_items:
        for item_title, status, explanation in compliance_items:
            status_map = {
                "pass": GREEN,
                "warning": AMBER,
                "fail": RED,
            }
            color = status_map.get(status, GRAY)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*color)
            icon = {"pass": "[OK]", "warning": "[!]", "fail": "[X]"}.get(status, "[ ]")
            pdf.cell(0, 6, f"  {icon}  {item_title}", new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*DARK)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(175, 4.5, f"     {explanation}")
            pdf.ln(2)

    # ── Section 6: Buyer Profile & Benchmarking ────────────────
    pdf.section_title(6, "Buyer Profile & Benchmarking")

    if buyer_profile:
        pdf.key_value_row("Total Procurements", str(buyer_profile.get("procurement_count", 0)))
        pdf.key_value_row("Price-Only Rate", f"{buyer_profile.get('price_only_rate', 0):.0%}")
        pdf.key_value_row("Single Bidder Rate", f"{buyer_profile.get('single_bidder_rate', 0):.0%}")
        pdf.key_value_row("Average Tenders", f"{buyer_profile.get('avg_tenders', 0):.1f}")
        pdf.key_value_row("VAKO Disputes", str(buyer_profile.get("vako_disputes", 0)))

        flags = buyer_profile.get("risk_flags", [])
        if flags:
            pdf.key_value_row("Risk Flags", ", ".join(flags))

        if sector_benchmarks and sector_benchmarks.get("total", 0) >= 10:
            pdf.ln(3)
            pdf.subsection(f"Buyer vs {sector_label} Sector Average")
            headers = ["Metric", "This Buyer", "Sector Average", "Assessment"]
            rows = []
            # Price-only comparison
            buyer_po = buyer_profile.get("price_only_rate", 0)
            sector_po = sector_benchmarks.get("price_only_rate", 0)
            assess = "Above avg" if buyer_po > sector_po + 0.1 else "Normal"
            rows.append(["Price-only rate", f"{buyer_po:.0%}", f"{sector_po:.0%}", assess])

            # Dispute rate
            dr = sector_benchmarks.get("dispute_rate", 0)
            rows.append(["Sector dispute rate", "-", f"{dr:.1%}", "Context"])

            pdf.simple_table(headers, rows, [45, 35, 40, 60])
    else:
        pdf.body_text("Buyer profile not available for this contracting authority.")

    # ── Section 7: Comparable Procurements ─────────────────────
    pdf.add_page()
    pdf.section_title(7, "Comparable Procurements")
    pdf.body_text(
        f"Past procurements in the same sector ({sector_label}) using "
        f"{procedure_label.lower()} procedure in a similar value range."
    )

    if comparable_procs:
        headers = ["Buyer", "Value", "Risk", "Disputed"]
        rows = []
        for cp in comparable_procs[:15]:
            rows.append([
                (cp.get("buyer", "") or "")[:25],
                cp.get("value_fmt", "-"),
                f"{cp.get('score', 0) * 100:.1f}%",
                "Yes" if cp.get("disputed") else "",
            ])
        pdf.simple_table(headers, rows, [65, 35, 30, 50])

        n_total = len(comparable_procs)
        n_disputed = sum(1 for c in comparable_procs if c.get("disputed"))
        if n_total:
            rate = n_disputed / n_total * 100
            pdf.body_text(
                f"Of {n_total} comparable procurements, {n_disputed} were disputed "
                f"({rate:.0f}% dispute rate)."
            )
    else:
        pdf.body_text("No comparable procurements found matching sector and procedure type.")

    # ── Section 8: Recommendations ─────────────────────────────
    pdf.section_title(8, "Recommendations")
    pdf.body_text(
        "Specific actions to reduce dispute risk, ordered by priority. "
        "Implementing these before publication can significantly reduce "
        "the likelihood of a VAKO challenge."
    )

    if checklist:
        for priority, action, rationale in checklist:
            pdf.recommendation_box(priority, action, rationale)
    else:
        pdf.info_box(
            "No specific recommendations at this time. This procurement follows "
            "standard patterns with low dispute risk.",
            "info"
        )

    # ── VAKO History (if any) ──────────────────────────────────
    if dispute_details:
        pdf.add_page()
        pdf.section_title(9, "VAKO Dispute History")
        for dd in dispute_details:
            pdf.key_value_row("Challenger", dd.get("challenger", "Unknown"))
            pdf.key_value_row("Submitted", dd.get("submitted", "Unknown"))
            pdf.key_value_row("Object", dd.get("object", "Unknown"))
            result = dd.get("result", dd.get("status", "Unknown"))
            pdf.key_value_row("Result", result, bold_value=True)
            pdf.ln(3)

    # ── Disclaimer ─────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title(10, "Methodology & Disclaimer")
    pdf.body_text(
        "This analysis was generated by the ProcureSight procurement risk intelligence "
        "system. The risk score is based on a logistic regression model trained on "
        "57,313 Estonian procurements from 2018-2026, validated against 959 real "
        "VAKO dispute decisions."
    )
    pdf.body_text(
        "The model uses 38 features including contract value, procedure type, sector, "
        "evaluation criteria weights, buyer track record, and integrity signals "
        "(political donor links, ownership concentration, company age, CPV price "
        "benchmarks, EU threshold proximity)."
    )
    pdf.body_text(
        "IMPORTANT: Risk scores are probabilistic indicators, not legal verdicts. "
        "A high risk score means the procurement resembles historically disputed "
        "ones along measured dimensions. It does not mean the procurement is "
        "non-compliant or will be challenged. Always consult qualified procurement "
        "specialists for final decisions."
    )
    pdf.body_text(
        "Data sources: riigihanked.riik.ee (procurement notices), VAKO decisions, "
        "e-Ariregister (company registry), ERJK (political financing records)."
    )

    return pdf.output()


def _build_compliance_review(features: dict, procedure: str, contract_type: str,
                              sector: str) -> list[tuple[str, str, str]]:
    """Build a legal compliance checklist. Returns [(title, status, explanation)]."""
    items = []
    pw = features.get("price_weight", 0)
    qw = features.get("quality_weight", 0)
    total_w = pw + qw
    value = 10 ** features.get("log_estimated_value", 0) if features.get("log_estimated_value", -1) > 0 else None

    # 1. Procedure appropriateness
    if procedure in ("neg-wo-call", "oth-single"):
        if value and value > 200_000:
            items.append((
                "Procedure selection (RHS 49)",
                "warning",
                "Non-competitive procedure used on a contract that may exceed simplified "
                "procedure thresholds. Verify that exemption grounds under RHS 28 are documented."
            ))
        else:
            items.append((
                "Procedure selection (RHS 49)",
                "pass",
                "Non-competitive procedure used within appropriate value range."
            ))
    elif procedure == "open":
        items.append((
            "Procedure selection (RHS 49)",
            "pass",
            "Open procedure ensures maximum competition and transparency."
        ))
    else:
        items.append((
            "Procedure selection (RHS 49)",
            "pass",
            f"Procedure type ({procedure}) noted. Verify appropriateness for contract scope."
        ))

    # 2. Evaluation criteria
    is_price_only = (qw == 0 and pw > 0) or (total_w > 0 and qw / total_w < 0.01)
    if is_price_only and contract_type == "services" and value and value > 200_000:
        items.append((
            "Evaluation criteria (RHS 85)",
            "warning",
            "Price-only evaluation on high-value services. EU guidelines recommend "
            "MEAT evaluation for complex contracts."
        ))
    elif pw == 0 and qw == 0:
        items.append((
            "Evaluation criteria (RHS 85)",
            "warning",
            "No evaluation criteria weights detected. Verify criteria are published."
        ))
    else:
        items.append((
            "Evaluation criteria (RHS 85)",
            "pass",
            f"Evaluation criteria defined (price {pw:.0%} / quality {qw:.0%})."
        ))

    # 3. Value publication
    if features.get("value_missing", 0):
        items.append((
            "Estimated value publication (RHS 23)",
            "warning",
            "Estimated contract value not published. Required for transparency."
        ))
    else:
        items.append((
            "Estimated value publication (RHS 23)",
            "pass",
            "Estimated value is published."
        ))

    # 4. Deadline adequacy
    deadline_days = features.get("log_deadline_days", -1)
    if deadline_days > 0:
        days = 10 ** deadline_days
        if days < 15 and procedure == "open":
            items.append((
                "Submission deadline (RHS 93)",
                "warning",
                f"Deadline of ~{days:.0f} days may be insufficient for open procedure. "
                "Minimum 35 days for open EU-level procedures."
            ))
        else:
            items.append((
                "Submission deadline (RHS 93)",
                "pass",
                f"Submission deadline of ~{days:.0f} days appears adequate."
            ))
    else:
        items.append((
            "Submission deadline (RHS 93)",
            "pass",
            "Deadline information not available for assessment."
        ))

    # 5. Strategic criteria
    has_any_strategic = (features.get("has_green") or features.get("has_social")
                         or features.get("has_innovation"))
    if has_any_strategic:
        strategic = []
        if features.get("has_green"):
            strategic.append("environmental")
        if features.get("has_social"):
            strategic.append("social")
        if features.get("has_innovation"):
            strategic.append("innovation")
        items.append((
            "Strategic procurement (RHS 77-79)",
            "pass",
            f"Strategic criteria included: {', '.join(strategic)}. "
            "This aligns with national procurement strategy goals."
        ))
    elif value and value > 500_000:
        items.append((
            "Strategic procurement (RHS 77-79)",
            "warning",
            "No strategic criteria (green, social, innovation) on a high-value contract. "
            "Consider whether sustainability or innovation criteria could apply."
        ))

    return items


def compute_quality_assessment(
    features: dict, procedure: str, contract_type: str,
    sector: str, buyer_profile: dict | None,
    integrity_flags: list,
) -> dict:
    """Compute a procurement quality score (0-100) across 5 dimensions.

    This is the core of the €20-30 value proposition — it tells the buyer
    not just "how risky" but "how good" their procurement is and exactly
    what to improve.
    """
    dimensions = {}
    pw = features.get("price_weight", 0)
    qw = features.get("quality_weight", 0)
    total_w = pw + qw

    # ── 1. Competition & Access (0-20) ─────────────────────────
    score_1 = 10  # Base: average
    finding_1 = ""
    if procedure == "open":
        score_1 += 5
        finding_1 = "Open procedure maximizes competition."
    elif procedure == "restricted":
        score_1 += 3
        finding_1 = "Restricted procedure: good if qualification stage is proportionate."
    elif procedure in ("neg-wo-call", "oth-single"):
        score_1 -= 8
        finding_1 = "Non-competitive procedure significantly limits access."
    elif procedure == "neg-w-call":
        score_1 -= 2
        finding_1 = "Negotiated procedure may limit competition."

    tenders = features.get("tenders_received", -1)
    if tenders >= 5:
        score_1 += 5
        finding_1 += f" {tenders} tenders received (excellent)."
    elif tenders >= 3:
        score_1 += 3
        finding_1 += f" {tenders} tenders received (adequate)."
    elif tenders == 1 and tenders != -1:
        score_1 -= 5
        finding_1 += " Only 1 tender received (poor competition)."

    score_1 = max(0, min(20, score_1))
    explanation_1 = (
        "Measures whether the procurement design encourages broad participation. "
        "Open procedures, reasonable deadlines, and proportionate qualifications "
        "attract more bidders and lead to better value for money. "
        + finding_1
    )
    dimensions["Competition & Access"] = {
        "score": score_1, "max": 20, "finding": finding_1,
        "explanation": explanation_1,
    }

    # ── 2. Criteria Quality (0-20) ─────────────────────────────
    score_2 = 10
    finding_2 = ""
    is_price_only = (qw == 0 and pw > 0) or (total_w > 0 and qw / total_w < 0.01)
    if is_price_only:
        if contract_type == "services":
            score_2 -= 6
            finding_2 = "Price-only evaluation on services contract risks quality degradation."
        else:
            score_2 -= 2
            finding_2 = "Price-only evaluation. Acceptable for standardized goods."
    elif total_w > 0 and qw / total_w >= 0.3:
        score_2 += 5
        finding_2 = f"Quality weight {qw / total_w:.0%} ensures value-for-money evaluation."
    elif total_w > 0 and qw / total_w >= 0.1:
        score_2 += 2
        finding_2 = f"Quality weight {qw / total_w:.0%} is modest but present."
    if pw == 0 and qw == 0:
        score_2 -= 4
        finding_2 = "No evaluation criteria weights published."

    score_2 = max(0, min(20, score_2))
    explanation_2 = (
        "Evaluates whether the award criteria are well-designed to select the best "
        "offer rather than just the cheapest. Quality criteria with clear scoring "
        "methodology reduce disputes and improve outcomes. "
        + finding_2
    )
    dimensions["Criteria Quality"] = {
        "score": score_2, "max": 20, "finding": finding_2,
        "explanation": explanation_2,
    }

    # ── 3. Strategic Value (0-20) ──────────────────────────────
    score_3 = 8
    finding_3 = ""
    strategic = []
    if features.get("has_green"):
        score_3 += 4
        strategic.append("environmental")
    if features.get("has_social"):
        score_3 += 4
        strategic.append("social")
    if features.get("has_innovation"):
        score_3 += 5
        strategic.append("innovation")

    if strategic:
        finding_3 = f"Strategic criteria: {', '.join(strategic)}."
    else:
        finding_3 = "No strategic criteria included."
        score_3 -= 3

    score_3 = max(0, min(20, score_3))
    explanation_3 = (
        "Assesses whether the procurement leverages public spending to advance "
        "policy goals (green transition, social inclusion, innovation). Estonia's "
        "national procurement strategy encourages these criteria. "
        + finding_3
    )
    dimensions["Strategic Value"] = {
        "score": score_3, "max": 20, "finding": finding_3,
        "explanation": explanation_3,
    }

    # ── 4. Transparency (0-20) ─────────────────────────────────
    score_4 = 12
    finding_4 = ""
    if not features.get("value_missing", 0):
        score_4 += 3
        finding_4 = "Value published."
    else:
        score_4 -= 4
        finding_4 = "Value not published."

    if not features.get("deadline_missing", 0):
        deadline_days = features.get("log_deadline_days", -1)
        if deadline_days > 0:
            days = 10 ** deadline_days
            if days >= 30:
                score_4 += 3
                finding_4 += f" Deadline {days:.0f} days (adequate)."
            elif days >= 15:
                score_4 += 1
                finding_4 += f" Deadline {days:.0f} days (short)."
            else:
                score_4 -= 3
                finding_4 += f" Deadline {days:.0f} days (very short)."

    score_4 = max(0, min(20, score_4))
    explanation_4 = (
        "Measures how well the procurement is documented and accessible. "
        "Published values, adequate deadlines, and clear specifications "
        "allow suppliers to make informed decisions about bidding. "
        + finding_4
    )
    dimensions["Transparency"] = {
        "score": score_4, "max": 20, "finding": finding_4,
        "explanation": explanation_4,
    }

    # ── 5. Integrity & Governance (0-20) ───────────────────────
    score_5 = 16  # High base: most procurements are clean
    finding_5 = ""
    if integrity_flags:
        for _, _, severity in integrity_flags:
            if severity in ("high", "HIGH"):
                score_5 -= 6
            elif severity in ("medium", "MEDIUM"):
                score_5 -= 3
        finding_5 = f"{len(integrity_flags)} integrity flag(s) identified."
    else:
        finding_5 = "No integrity concerns identified."

    if buyer_profile:
        if buyer_profile.get("vako_disputes", 0) >= 3:
            score_5 -= 3
            finding_5 += " Buyer has multiple past disputes."
        if buyer_profile.get("single_bidder_rate", 0) > 0.5:
            score_5 -= 2
            finding_5 += " Buyer has high single-bidder rate."

    score_5 = max(0, min(20, score_5))
    explanation_5 = (
        "Evaluates the integrity environment around the procurement: political "
        "connections, ownership structures, buyer track record, and price benchmarks. "
        + finding_5
    )
    dimensions["Integrity & Governance"] = {
        "score": score_5, "max": 20, "finding": finding_5,
        "explanation": explanation_5,
    }

    # ── Overall ────────────────────────────────────────────────
    overall = sum(d["score"] for d in dimensions.values())
    if overall >= 80:
        summary = "This procurement demonstrates strong practices across all dimensions."
    elif overall >= 60:
        summary = "Good foundation with specific areas for improvement identified below."
    elif overall >= 40:
        summary = "Several dimensions need attention. Review recommendations carefully."
    else:
        summary = "Significant improvements needed across multiple dimensions."

    return {
        "overall_score": overall,
        "summary": summary,
        "dimensions": dimensions,
    }
