"""
Procurement Risk Monitor — Demo Dashboard
==========================================
Streamlit app showcasing the Estonian procurement risk scoring pipeline.

Usage:
    cd procurement-monitor
    python3 -m streamlit run scripts/demo_app.py
"""

from __future__ import annotations

import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure pdf_report is importable from both dev and deploy layouts:
#   Dev:    scripts/demo_app.py + scripts/pdf_report.py  (same dir)
#   Deploy: app.py + scripts/pdf_report.py               (scripts/ subdir)
_this_dir = Path(__file__).resolve().parent
for _candidate in [_this_dir, _this_dir / "scripts"]:
    _candidate_str = str(_candidate)
    if _candidate_str not in sys.path:
        sys.path.insert(0, _candidate_str)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODEL_DIR = DATA / "model"
V3_DIR = DATA / "adversarial_v3"

# ---------------------------------------------------------------------------
# Data loading (all cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_monthly_results(month: str) -> dict | None:
    path = MODEL_DIR / f"stage1_results_{month}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_combined_results(month: str) -> dict | None:
    path = MODEL_DIR / f"combined_results_{month}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_all_scores() -> pd.DataFrame:
    with open(MODEL_DIR / "stage1_scores.json") as f:
        data = json.load(f)
    return pd.DataFrame(data)


@st.cache_data
def load_features() -> dict:
    """Return dict keyed by rhr_id -> feature record."""
    with open(MODEL_DIR / "features.json") as f:
        data = json.load(f)
    return {r["rhr_id"]: r for r in data}


@st.cache_data
def load_buyer_profiles() -> dict:
    with open(DATA / "v1_results" / "buyer_profiles.json") as f:
        data = json.load(f)
    return {r["buyer_name"]: r for r in data}


@st.cache_data
def load_disputes() -> dict:
    with open(DATA / "ground_truth" / "vako_disputes.json") as f:
        return json.load(f)


@st.cache_data
def load_model():
    with open(MODEL_DIR / "stage1_model.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_v3_results() -> dict | None:
    path = V3_DIR / "v3_results.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_procurement_titles() -> dict:
    path = MODEL_DIR / "procurement_titles.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_integrity_lookups() -> dict:
    path = MODEL_DIR / "integrity_lookups.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_gap_analysis() -> dict:
    path = DATA / "gap_analysis_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_phase2_results() -> dict:
    path = DATA / "phase2_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_enriched_procurements() -> dict:
    """Load enriched procurement data keyed by rhr_id."""
    path = MODEL_DIR / "enriched_procurements.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {str(r["rhr_id"]): r for r in data}


def load_extracted_text(rhr_id: str) -> str | None:
    """Load full extracted document text for a v3 procurement."""
    path = V3_DIR / "extracted_text" / f"{rhr_id}.txt"
    if not path.exists():
        return None
    with open(path) as f:
        return f.read()


def available_months() -> list[str]:
    months = []
    for p in sorted(MODEL_DIR.glob("stage1_results_*.json")):
        m = p.stem.replace("stage1_results_", "")
        months.append(m)
    return months


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_eur(v):
    if v is None or pd.isna(v):
        return "\u2014"
    if v >= 1_000_000:
        return f"\u20ac{v / 1_000_000:,.1f}M"
    if v >= 1_000:
        return f"\u20ac{v / 1_000:,.0f}K"
    return f"\u20ac{v:,.0f}"


# ---------------------------------------------------------------------------
# Internationalisation (EN / ET)
# ---------------------------------------------------------------------------
TRANSLATIONS = {
    # -- Page names --
    "page_about": {"en": "About", "et": "Tutvustus"},
    "page_risk_monitor": {"en": "Live Risk Monitor", "et": "Riskimonitor"},
    "page_compliance": {"en": "Compliance Scan", "et": "Nõuetele vastavus"},
    "page_integrity": {"en": "Integrity Analysis", "et": "Aususe analüüs"},
    "page_historical": {"en": "Historical Analysis", "et": "Ajalugu"},
    "page_deep_dive": {"en": "Procurement Deep Dive", "et": "Hanke süvaanalüüs"},
    # -- Sidebar --
    "sidebar_subtitle": {"en": "Procurement Risk Intelligence", "et": "Riigihanke riskianalüüs"},
    "sidebar_cta_title": {"en": "Analyse your procurement", "et": "Analüüsi oma hanget"},
    "sidebar_cta_body": {
        "en": "Go to <strong>Deep Dive</strong> to get a full quality assessment with downloadable PDF report.",
        "et": "Ava <strong>Süvaanalüüs</strong>, et saada täielik kvaliteedihinnang allalaaditava PDF-raportiga.",
    },
    "sidebar_procurements": {"en": "Procurements", "et": "Hanked"},
    "sidebar_disputes": {"en": "Disputes", "et": "Vaidlustused"},
    "sidebar_data": {"en": "Data", "et": "Andmed"},
    "sidebar_source": {"en": "Source", "et": "Allikas"},
    # -- Risk tiers --
    "tier_low": {"en": "Low", "et": "Madal"},
    "tier_moderate": {"en": "Moderate", "et": "Mõõdukas"},
    "tier_elevated": {"en": "Elevated", "et": "Kõrgendatud"},
    "tier_high": {"en": "High", "et": "Kõrge"},
    # -- Live Risk Monitor --
    "lrm_title": {"en": "Live Risk Monitor", "et": "Riskimonitor"},
    "lrm_procurements": {"en": "Procurements", "et": "Hanked"},
    "lrm_elevated_risk": {"en": "Elevated Risk", "et": "Kõrgendatud risk"},
    "lrm_high_risk": {"en": "High Risk", "et": "Kõrge risk"},
    "lrm_known_disputes": {"en": "Known VAKO Disputes", "et": "Teadaolevad VAKO vaidlustused"},
    "lrm_of_total": {"en": "of total", "et": "koguarvust"},
    "lrm_total_value": {"en": "Total Value", "et": "Koguväärtus"},
    "lrm_median_value": {"en": "Median Value", "et": "Mediaanväärtus"},
    "lrm_common_proc": {"en": "Most Common Procedure", "et": "Levinuim menetlus"},
    "lrm_top_sector": {"en": "Top Sector", "et": "Enim hanked"},
    "lrm_risk_distribution": {"en": "Risk Score Distribution", "et": "Riskiskooride jaotus"},
    "lrm_risk_by_sector": {"en": "Average Risk by Sector", "et": "Keskmine risk sektori järgi"},
    "lrm_by_risk_score": {"en": "Procurements by Risk Score", "et": "Hanked riskiskoori järgi"},
    "lrm_show": {"en": "Show", "et": "Näita"},
    "lrm_risk_level": {"en": "Risk Level", "et": "Riskitase"},
    "lrm_sector": {"en": "Sector", "et": "Sektor"},
    "lrm_all": {"en": "All", "et": "Kõik"},
    "lrm_click_row": {"en": "Click any row to open full analysis", "et": "Klõpsa real, et avada täisanalüüs"},
    "lrm_disputed": {"en": "DISPUTED", "et": "VAIDLUSTATUD"},
    # -- Table columns --
    "col_rank": {"en": "#", "et": "#"},
    "col_risk": {"en": "Risk", "et": "Risk"},
    "col_level": {"en": "Level", "et": "Tase"},
    "col_procurement": {"en": "Procurement", "et": "Hange"},
    "col_buyer": {"en": "Buyer", "et": "Hankija"},
    "col_sector": {"en": "Sector", "et": "Sektor"},
    "col_procedure": {"en": "Procedure", "et": "Menetlus"},
    "col_value": {"en": "Value", "et": "Väärtus"},
    "col_status": {"en": "Status", "et": "Staatus"},
    # -- Deep Dive --
    "dd_title": {"en": "Procurement Deep Dive", "et": "Hanke süvaanalüüs"},
    "dd_back": {"en": "\u2190 Back to Risk Monitor", "et": "\u2190 Tagasi riskimonitori"},
    "dd_top_flagged": {"en": "Top Flagged", "et": "Kõrgeima riskiga"},
    "dd_search_all": {"en": "Search All", "et": "Otsi kõigist"},
    "dd_search_placeholder": {"en": "Search by title, buyer, or RHR ID", "et": "Otsi pealkirja, hankija või RHR ID järgi"},
    "dd_risk_score": {"en": "Risk Score", "et": "Riskiskoor"},
    "dd_feature_contrib": {"en": "Feature Contributions", "et": "Riskitegurite panus"},
    "dd_risk_summary": {"en": "Risk Assessment Summary", "et": "Riskihinnangu kokkuvõte"},
    "dd_quality_score": {"en": "Procurement Quality Score", "et": "Hanke kvaliteediskoor"},
    "dd_quality_breakdown": {"en": "View detailed quality breakdown", "et": "Vaata detailset kvaliteedianalüüsi"},
    "dd_dispute_details": {"en": "VAKO Dispute Details", "et": "VAKO vaidlustuse üksikasjad"},
    "dd_integrity_flags": {"en": "Integrity Flags", "et": "Aususe märgised"},
    "dd_checklist": {"en": "Pre-Publication Checklist", "et": "Avaldamiseelne kontrollnimekiri"},
    "dd_buyer_profile": {"en": "Buyer Profile", "et": "Hankija profiil"},
    "dd_ai_analysis": {"en": "AI Document Analysis", "et": "AI dokumendianalüüs"},
    "dd_comparable": {"en": "Comparable Procurements", "et": "Võrreldavad hanked"},
    "dd_download_title": {"en": "Download Full Report", "et": "Laadi alla täisraport"},
    "dd_download_desc": {
        "en": "Professional PDF with quality assessment, integrity checks, legal compliance review, buyer benchmarking, comparable procurements, and recommendations.",
        "et": "Professionaalne PDF kvaliteedihinnanguga, aususe kontrollidega, õiguslike vastavuste ülevaatega, hankija võrdlusanalüüsiga ja soovitustega.",
    },
    "dd_download_btn": {"en": "Download Full Analysis (PDF)", "et": "Laadi alla täisanalüüs (PDF)"},
    # -- Quality dimensions --
    "qdim_competition": {"en": "Competition & Access", "et": "Konkurents ja juurdepääs"},
    "qdim_criteria": {"en": "Criteria Quality", "et": "Kriteeriumide kvaliteet"},
    "qdim_strategy": {"en": "Strategic Value", "et": "Strateegiline väärtus"},
    "qdim_transparency": {"en": "Transparency", "et": "Läbipaistvus"},
    "qdim_integrity": {"en": "Integrity & Governance", "et": "Ausus ja valitsemine"},
    # -- Quality labels --
    "quality_excellent": {"en": "Excellent", "et": "Suurepärane"},
    "quality_good": {"en": "Good", "et": "Hea"},
    "quality_fair": {"en": "Fair", "et": "Rahuldav"},
    "quality_needs_work": {"en": "Needs Improvement", "et": "Vajab parandamist"},
    # -- About page --
    "about_hero_subtitle": {"en": "Procurement Risk Intelligence", "et": "Riigihanke riskianalüüs"},
    "about_hero_text": {
        "en": "Check your procurement <strong>before you publish</strong>. Analyses your tender against <strong>57,000+ historical procurements</strong> and <strong>959 real VAKO disputes</strong> to find compliance risks, suggest improvements, and help you avoid costly challenges.",
        "et": "Kontrolli oma hanget <strong>enne avaldamist</strong>. Analüüsib sinu hankedokumente <strong>57 000+ varasema hanke</strong> ja <strong>959 tegeliku VAKO vaidlustuse</strong> põhjal, et leida nõuetele vastavuse riske, soovitada parandusi ja aidata vältida kulukaid vaidlustusi.",
    },
    "about_dispute_cost": {"en": "Average VAKO dispute cost", "et": "Keskmine VAKO vaidlustuse kulu"},
    "about_analysis_cost": {"en": "analysis", "et": "analüüs"},
    "about_procs_analysed": {"en": "Procurements Analysed", "et": "Analüüsitud hanked"},
    "about_disputes_studied": {"en": "VAKO Disputes Studied", "et": "Uuritud VAKO vaidlustused"},
    "about_risk_signals": {"en": "Risk Signals Checked", "et": "Kontrollitud riskisignaalid"},
    "about_coverage": {"en": "Data Coverage", "et": "Andmete katvus"},
    "about_what_you_get": {"en": "What you get", "et": "Mida saad"},
    "about_prevent": {"en": "Prevent Disputes", "et": "Väldi vaidlustusi"},
    "about_improve": {"en": "Improve Quality", "et": "Paranda kvaliteeti"},
    "about_ensure": {"en": "Ensure Integrity", "et": "Taga ausus"},
    "about_how_works": {"en": "How it works", "et": "Kuidas see töötab"},
    "about_explore": {"en": "Explore the dashboard", "et": "Tutvu töölauaga"},
    "about_data_sources": {"en": "Data sources", "et": "Andmeallikad"},
    "about_how_to_read": {"en": "How to read the results", "et": "Kuidas tulemusi lugeda"},
    # -- Compliance --
    "comp_title": {"en": "Compliance Scan", "et": "Nõuetele vastavuse kontroll"},
    "comp_intro": {
        "en": "Specific rule violations and data quality issues found in this month's procurements. These are **definite issues** \u2014 not probabilistic scores but concrete findings that can be verified.",
        "et": "Selle kuu hangetes leitud konkreetsed rikkumised ja andmekvaliteedi probleemid. Need on **kindlad leiud** \u2014 mitte tõenäosuslikud skoorid, vaid kontrollitavad faktid.",
    },
    "comp_critical": {"en": "Critical", "et": "Kriitiline"},
    "comp_warning": {"en": "Warning", "et": "Hoiatus"},
    "comp_info": {"en": "Info", "et": "Info"},
    # -- Integrity --
    "int_title": {"en": "Integrity Analysis", "et": "Aususe analüüs"},
    "int_intro": {
        "en": "Automated cross-referencing of procurement winners against external registries: political donor records, company registry, and ownership data. These checks run on **all 57,000+ procurements** automatically.",
        "et": "Automaatne hangete võitjate ristkontroll väliste registritega: erakondade rahastamine, äriregister ja omandiandmed. Need kontrollid tehakse **kõigile 57 000+ hanketele** automaatselt.",
    },
    "int_political_donors": {"en": "Political Donors", "et": "Poliitilised annetajad"},
    "int_hidden_ownership": {"en": "Hidden Ownership", "et": "Varjatud omand"},
    "int_near_threshold": {"en": "Near EU Threshold", "et": "EL piirmäära lähedal"},
    "int_young_winners": {"en": "Young Winners", "et": "Noored võitjad"},
    "int_price_anomalies": {"en": "Price Anomalies", "et": "Hinna anomaaliad"},
    # -- Historical --
    "hist_title": {"en": "Historical Analysis", "et": "Ajalooline analüüs"},
    "hist_subtitle": {
        "en": "Patterns and trends across **57,000+ procurements** from 2018 to 2026.",
        "et": "Mustrid ja trendid **57 000+ hankes** aastatel 2018\u20132026.",
    },
    "hist_key_findings": {"en": "Key Findings", "et": "Peamised leiud"},
    "hist_monthly_volume": {"en": "Monthly Procurement Volume & Disputes", "et": "Igakuine hangete arv ja vaidlustused"},
    "hist_by_value": {"en": "Dispute Rate by Contract Value", "et": "Vaidlustuse määr lepingu väärtuse järgi"},
    "hist_by_sector": {"en": "Dispute Rate by Sector", "et": "Vaidlustuse määr sektori järgi"},
    "hist_by_procedure": {"en": "Dispute Rate by Procedure Type", "et": "Vaidlustuse määr menetluse liigi järgi"},
    "hist_top_buyers": {"en": "Top 20 Riskiest Buyers", "et": "20 kõrgeima riskiga hankijat"},
}


def t(key: str) -> str:
    """Return translated string for current language."""
    lang = st.session_state.get("lang", "en")
    entry = TRANSLATIONS.get(key)
    if not entry:
        return key
    return entry.get(lang, entry.get("en", key))


SECTOR_LABELS_EN = {
    "IT": "IT & Digital",
    "construction": "Construction",
    "transport": "Transport",
    "energy": "Energy",
    "healthcare": "Healthcare",
    "consulting": "Consulting",
    "professional_services": "Professional Services",
    "maintenance": "Maintenance",
    "education": "Education",
    "environment": "Environment",
    "defense": "Defence",
    "other": "Other",
}

SECTOR_LABELS_ET = {
    "IT": "IT ja digitaal",
    "construction": "Ehitus",
    "transport": "Transport",
    "energy": "Energeetika",
    "healthcare": "Tervishoid",
    "consulting": "Nõustamine",
    "professional_services": "Eriteenused",
    "maintenance": "Hooldus",
    "education": "Haridus",
    "environment": "Keskkond",
    "defense": "Riigikaitse",
    "other": "Muu",
}

PROCEDURE_LABELS_EN = {
    "open": "Open",
    "restricted": "Restricted",
    "neg-w-call": "Negotiated (w/ call)",
    "neg-wo-call": "Negotiated (w/o call)",
    "oth-single": "Single source",
    "simplified": "Simplified",
    "concession": "Concession",
}

PROCEDURE_LABELS_ET = {
    "open": "Avatud",
    "restricted": "Piiratud",
    "neg-w-call": "Läbirääkimistega (väljakuulutamisega)",
    "neg-wo-call": "Läbirääkimistega (ilma väljakuulutamiseta)",
    "oth-single": "Otseost",
    "simplified": "Lihtsustatud",
    "concession": "Kontsessioon",
}


def _sector_labels() -> dict:
    return SECTOR_LABELS_ET if st.session_state.get("lang") == "et" else SECTOR_LABELS_EN


def _procedure_labels() -> dict:
    return PROCEDURE_LABELS_ET if st.session_state.get("lang") == "et" else PROCEDURE_LABELS_EN


# Dynamic label accessors are _sector_labels() and _procedure_labels()

# Explanations for each feature contribution shown in the deep dive
FEATURE_EXPLANATIONS = {
    "log_estimated_value": (
        "Contract Value",
        "Higher-value contracts attract more scrutiny and are more likely to be "
        "challenged. This is the single strongest predictor: contracts above \u20ac5M "
        "have a 22% dispute rate vs near-zero for contracts under \u20ac1M."
    ),
    "value_missing": (
        "Value Missing",
        "When the estimated contract value is not published, this strongly "
        "increases risk. Missing values may indicate either very large contracts "
        "or incomplete documentation."
    ),
    "proc_open": (
        "Open Procedure",
        "Open procedures attract more bidders and more potential challengers. "
        "While open is the most transparent procedure, it also generates the "
        "most disputes simply because more companies participate."
    ),
    "proc_restricted": (
        "Restricted Procedure",
        "Restricted procedures limit who can bid via a pre-qualification stage. "
        "Moderate risk: fewer bidders but selection criteria can be challenged."
    ),
    "proc_neg-w-call": (
        "Negotiated (w/ call)",
        "Negotiated procedures with a prior call for competition. Historically "
        "carries elevated risk (18.2% dispute rate) because the negotiation "
        "process and qualification criteria are common dispute targets."
    ),
    "proc_neg-wo-call": (
        "Negotiated (w/o call)",
        "Negotiated without competition \u2014 essentially sole sourcing. Rarely "
        "disputed because there are few participants to challenge it, but may "
        "indicate compliance issues."
    ),
    "proc_oth-single": (
        "Single Source",
        "Direct award to a single supplier. Low dispute volume but often "
        "flags compliance concerns about why competition was bypassed."
    ),
    "tenders_missing": (
        "Tenders Missing",
        "Number of bids received is not yet recorded. Common for ongoing "
        "procurements. Increases uncertainty in the model."
    ),
    "tenders_received": (
        "Tenders Received",
        "More tenders means more potential challengers. Procurements with "
        "many bidders see more disputes, especially from losing bidders."
    ),
    "log_deadline_days": (
        "Deadline Length",
        "Longer submission deadlines slightly increase dispute probability. "
        "This may proxy for contract complexity rather than being causal."
    ),
    "deadline_missing": (
        "Deadline Missing",
        "Submission deadline not recorded. Minimal predictive impact."
    ),
    "price_weight": (
        "Price Weight",
        "The weight given to price in evaluation. Surprisingly, pure price "
        "evaluation is NOT a strong dispute predictor \u2014 disputes are more common "
        "when complex quality criteria create ambiguity."
    ),
    "quality_weight": (
        "Quality Weight",
        "The weight given to quality criteria. Higher quality weights can "
        "increase disputes because subjective criteria are easier to challenge."
    ),
    "ct_services": (
        "Services Contract",
        "Service contracts have moderate dispute risk. Harder to specify "
        "precisely than goods, creating more room for challenges."
    ),
    "ct_supplies": (
        "Supplies Contract",
        "Supply contracts (goods) have lower dispute rates. Specifications "
        "are more objective and easier to evaluate."
    ),
    "ct_works": (
        "Works Contract",
        "Construction/works contracts. Risk depends heavily on value \u2014 "
        "large infrastructure projects are heavily disputed."
    ),
    "sector_IT": (
        "IT Sector",
        "IT procurement has a 3.7% dispute rate. Technology specifications "
        "can be restrictive or favour specific vendors."
    ),
    "sector_construction": (
        "Construction Sector",
        "Construction has a relatively low 1.5% dispute rate despite high "
        "values, likely because the sector has mature procurement practices."
    ),
    "sector_transport": (
        "Transport Sector",
        "Transport procurement has a 5.1% dispute rate \u2014 above average. "
        "Large infrastructure contracts attract scrutiny."
    ),
    "sector_energy": (
        "Energy Sector",
        "Energy has the highest dispute rate at 7.7%. Concessions and large "
        "infrastructure investments create complex procurement situations."
    ),
    "sector_healthcare": (
        "Healthcare Sector",
        "Healthcare procurement has moderate risk. Medical specifications "
        "can inadvertently favour specific manufacturers."
    ),
    "sector_consulting": (
        "Consulting Sector",
        "Consulting services have lower dispute rates. The subjective nature "
        "of evaluation is accepted in this sector."
    ),
    "sector_professional_services": (
        "Professional Services",
        "Professional services (legal, audit, etc.) have below-average "
        "dispute rates."
    ),
    "has_green": (
        "Green Criteria",
        "Procurement includes environmental sustainability criteria. "
        "Minimal direct impact on dispute risk."
    ),
    "has_social": (
        "Social Criteria",
        "Procurement includes social responsibility criteria. "
        "Slightly reduces dispute probability."
    ),
    "has_innovation": (
        "Innovation Criteria",
        "Procurement includes innovation criteria. "
        "Slightly increases risk as novel requirements can be ambiguous."
    ),
    "is_eu_funded": (
        "EU Funded",
        "Procurement co-funded by the EU. Slightly higher risk because "
        "EU-funded projects follow stricter rules and attract more oversight."
    ),
    "is_framework": (
        "Framework Agreement",
        "Framework agreements cover multiple future orders. Minimal impact "
        "on dispute probability."
    ),
    "buyer_procurement_count": (
        "Buyer Experience",
        "How many procurements this buyer has conducted. More experienced "
        "buyers (higher count) tend to have fewer disputes."
    ),
    "buyer_price_only_rate": (
        "Buyer Price-Only Rate",
        "How often this buyer uses price-only evaluation. Counterintuitively, "
        "this slightly reduces risk \u2014 simple criteria leave less room for disputes."
    ),
    "buyer_risk_score": (
        "Buyer Risk Score",
        "Aggregate risk score for this buyer based on historical patterns. "
        "Has a small coefficient because it is partially circular with outcomes."
    ),
    "buyer_missing": (
        "Unknown Buyer",
        "Buyer identity could not be matched. Minimal impact."
    ),
    "cpv_price_zscore": (
        "CPV Price Anomaly",
        "How unusual this contract value is compared to others in the same "
        "CPV-4 category (z-score). Values above 2\u03c3 indicate the contract "
        "is significantly more expensive than typical for its category."
    ),
    "threshold_proximity": (
        "Threshold Proximity",
        "Whether the contract value falls in the 90-99% band just below an EU "
        "procurement threshold (\u20ac143K, \u20ac443K, \u20ac5.5M). This pattern can indicate "
        "deliberate threshold avoidance."
    ),
    "winner_age_years": (
        "Company Age",
        "How old the winning company was at the time of contract award. Very "
        "young companies (under 2 years) winning large contracts may indicate "
        "shell companies or front entities."
    ),
    "winner_age_missing": (
        "Company Age Unknown",
        "Winning company's registration date could not be found in the "
        "business registry. Common for foreign or unmatched entities."
    ),
    "donor_linked": (
        "Political Donor Link",
        "Whether the winning company has a board member who donated \u20ac5K+ to a "
        "political party. Based on ERJK (party financing) records cross-referenced "
        "with e-Business Registry board member data. Companies with material donor "
        "links have a 2.7x higher dispute rate."
    ),
    "hidden_concentration": (
        "Hidden Ownership Concentration",
        "Whether the winning company shares a beneficial owner with another company "
        "that also won contracts from the same buyer, but under a different name. "
        "This can indicate undisclosed related-party transactions."
    ),
}


def risk_color(score: float) -> str:
    if score >= 0.15:
        return "#d32f2f"
    if score >= 0.08:
        return "#f57c00"
    if score >= 0.04:
        return "#fbc02d"
    return "#388e3c"


def risk_label(score: float) -> str:
    if score >= 0.15:
        return "High"
    if score >= 0.08:
        return "Elevated"
    if score >= 0.04:
        return "Moderate"
    return "Low"


# Compliance rule definitions: id -> (title, explanation, severity)
COMPLIANCE_RULES = {
    "non_competitive_high_value": (
        "Non-competitive procedure on high-value contract",
        "The Public Procurement Act (RHS \u00a7 49) requires competitive procedures "
        "(open, restricted) for contracts above EU thresholds (\u20ac143K services/supplies, "
        "\u20ac5.5M works). Single-source or negotiated-without-call procedures are only "
        "permitted under specific exemptions (RHS \u00a7 28), which must be documented. "
        "This rule flags contracts above \u20ac200K using non-competitive procedures.",
        "high",
    ),
    "single_source_works_threshold": (
        "Single-source works above EU threshold",
        "Works contracts above \u20ac5.5M must use open or restricted procedure under EU "
        "Directive 2014/24/EU unless narrow exemptions apply (extreme urgency, "
        "exclusive rights). This is a serious compliance issue.",
        "high",
    ),
    "price_only_high_services": (
        "Price-only evaluation on high-value services",
        "EU Directive 2014/24/EU and RHS guidelines recommend MEAT (most economically "
        "advantageous tender) evaluation for complex services. Price-only evaluation "
        "above \u20ac500K risks selecting underqualified providers. Quality criteria such as "
        "methodology, team experience, or delivery approach help ensure value for money.",
        "medium",
    ),
    "no_criteria": (
        "No evaluation criteria specified",
        "RHS \u00a7 85 requires evaluation criteria and their relative weighting to be "
        "published in procurement documents. Missing criteria make the award "
        "decision non-transparent and vulnerable to challenge.",
        "high",
    ),
    "low_competition_buyer": (
        "Buyer has persistently low competition",
        "When a buyer consistently receives single bids (\u226560% of procurements), "
        "this may indicate restrictive specifications, insufficient market outreach, "
        "overly short deadlines, or de facto vendor lock-in. The European Commission "
        "considers single-bidder rates above 30% a red flag.",
        "medium",
    ),
    "price_only_disputed_buyer": (
        "Buyer uses price-only despite repeat disputes",
        "Buyers with multiple VAKO disputes who still use 100% price-only evaluation "
        "may not be adapting their procurement practices. Diversifying criteria can "
        "reduce ambiguity and challenge risk.",
        "medium",
    ),
    "brand_name_restriction": (
        "Possible brand-name or vendor-specific restriction",
        "RHS \u00a7 87(6) prohibits technical specifications that refer to a specific make, "
        "source, or process in a way that favours or eliminates certain companies, unless "
        "accompanied by 'or equivalent'. Brand-specific requirements are among the most "
        "commonly sustained VAKO challenges. This rule scans procurement titles for "
        "product names, vendor names, or platform-specific references.",
        "high",
    ),
    "missing_buyer": (
        "Missing buyer name",
        "The contracting authority must be identified in all procurement notices. "
        "Missing buyer name indicates a data quality issue in the source XML.",
        "low",
    ),
}

# Patterns that suggest brand/vendor-specific procurement
BRAND_PATTERNS = [
    # Software vendors
    r"\bSAP\b", r"\bOracle\b", r"\bMicrosoft\b", r"\bSalesforce\b",
    r"\bVMware\b", r"\bCisco\b", r"\bIBM\b", r"\bAdobe\b",
    r"\bAmazon\b", r"\bAWS\b", r"\bGoogle Cloud\b", r"\bAzure\b",
    # Hardware
    r"\bApple\b", r"\bDell\b", r"\bHP\b", r"\bLenovo\b",
    # Specific platforms/products
    r"\bSharePoint\b", r"\bDynamics\b", r"\b365\b",
    r"\bAutoCAD\b", r"\bArcGIS\b", r"\bQlik\b", r"\bTableau\b",
    # Estonian-specific patterns for vendor lock
    r"\blitsents", r"\bhooldus.*leping",
]


def detect_clear_errors(df: pd.DataFrame, buyer_profiles: dict,
                        titles_data: dict) -> pd.DataFrame:
    """Flag procurements with clear compliance or data issues."""
    flags = []
    for _, row in df.iterrows():
        issues = []
        pw = row.get("price_weight", 0) or 0
        qw = row.get("quality_weight", 0) or 0
        val = row.get("estimated_value")
        proc = row.get("procedure_type", "")
        ctype = row.get("contract_type", "")
        buyer = row.get("buyer_name", "")
        rhr_id = row.get("rhr_id", "")

        # Get title for brand-name scanning
        title_text = titles_data.get(str(rhr_id), {}).get("title", "")

        # --- Procedure issues ---
        if proc in ("neg-wo-call", "oth-single") and val and val > 200_000:
            rule = COMPLIANCE_RULES["non_competitive_high_value"]
            issues.append(("non_competitive_high_value",
                           f"Value {fmt_eur(val)} awarded via "
                           f"{_procedure_labels().get(proc, proc)}. "
                           "Contracts above EU thresholds normally require competitive "
                           "procedures. Justification should be documented (RHS \u00a7 28)."))

        if proc in ("neg-wo-call", "oth-single") and ctype == "works" and val and val > 5_000_000:
            issues.append(("single_source_works_threshold",
                           f"Works contract worth {fmt_eur(val)} without competition. "
                           "This exceeds the EU threshold for works."))

        # --- Evaluation criteria issues ---
        total_w = pw + qw
        is_price_only = (qw == 0 and pw > 0) or (total_w > 0 and qw / total_w < 0.01)
        if is_price_only and val and val > 500_000 and ctype == "services":
            issues.append(("price_only_high_services",
                           f"Services contract worth {fmt_eur(val)} evaluated on "
                           "price alone. Quality criteria recommended for complex services."))

        if pw == 0 and qw == 0 and proc in ("open", "restricted", "neg-w-call"):
            issues.append(("no_criteria",
                           "Neither price nor quality weights are defined for a "
                           "competitive procedure."))

        # --- Brand-name / vendor-specific scan ---
        if title_text:
            for pattern in BRAND_PATTERNS:
                m = re.search(pattern, title_text, re.IGNORECASE)
                if m:
                    matched_text = m.group(0)
                    issues.append(("brand_name_restriction",
                                   f"Procurement title contains '{matched_text}': "
                                   f"\"{title_text[:120]}\" \u2014 "
                                   "If this refers to a specific product or vendor, the "
                                   "specification should include 'or equivalent' language."))
                    break  # one brand flag per procurement

        # --- Buyer pattern issues ---
        profile = buyer_profiles.get(buyer)
        if profile and profile.get("procurement_count", 0) >= 5:
            sbr = profile.get("single_bidder_rate", 0)
            if sbr >= 0.6 and profile.get("procurement_count", 0) >= 10:
                issues.append(("low_competition_buyer",
                               f"{buyer} receives a single bid in "
                               f"{sbr:.0%} of procurements "
                               f"(across {profile['procurement_count']} total)."))

            if profile.get("price_only_rate", 0) >= 0.95 and profile.get("vako_disputes", 0) >= 2:
                issues.append(("price_only_disputed_buyer",
                               f"{buyer}: {profile['price_only_rate']:.0%} price-only "
                               f"with {profile['vako_disputes']} VAKO disputes."))

        # --- Data quality ---
        if not buyer:
            issues.append(("missing_buyer", "Contracting authority not identified."))

        if issues:
            for rule_id, detail in issues:
                rule_title = COMPLIANCE_RULES.get(rule_id, (rule_id, "", "medium"))[0]
                flags.append({
                    "rhr_id": rhr_id,
                    "buyer_name": buyer,
                    "sector": row.get("sector", ""),
                    "procedure_type": proc,
                    "estimated_value": val,
                    "stage1_probability": row.get("stage1_probability", 0),
                    "rule_id": rule_id,
                    "issue_title": rule_title,
                    "issue_detail": detail,
                })
    return pd.DataFrame(flags) if flags else pd.DataFrame()


# ---------------------------------------------------------------------------
# Deep dive helpers (must be defined before page logic runs)
# ---------------------------------------------------------------------------

def _clean_title(raw_title: str, max_len: int = 120) -> str:
    """Extract a clean short title from the raw XML description."""
    if not raw_title:
        return ""
    # Remove "Hanke objektiks on" prefix (common Estonian procurement boilerplate)
    text = re.sub(r"^Hanke objekt[a-z]* on\s+", "", raw_title, flags=re.IGNORECASE)
    # Try to cut at first sentence boundary
    for sep in [". ", ".\n", " ning ", " ja "]:
        pos = text.find(sep)
        if 30 < pos < max_len:
            return text[:pos].strip()
    # Fall back to truncating at max_len on word boundary
    if len(text) <= max_len:
        return text.strip()
    truncated = text[:max_len].rsplit(" ", 1)[0]
    return truncated.rstrip(".,;:") + "..."


def _get_procurement_title(rhr_id: str, v3_lookup: dict, disputes_data: dict,
                           titles_data: dict) -> tuple[str, str]:
    """Look up a procurement title. Returns (short_title, full_title)."""
    raw = ""
    # Titles file (extracted from XML) has broadest coverage
    t = titles_data.get(rhr_id, {})
    if t.get("title"):
        raw = t["title"]
    if not raw:
        v3_rec = v3_lookup.get(rhr_id)
        if v3_rec:
            title = v3_rec.get("title", "")
            if title and title != "Unknown":
                raw = title
    if not raw:
        disp_rec = disputes_data.get("disputes", {}).get(rhr_id)
        if disp_rec:
            title = disp_rec.get("procurement", {}).get("title", "")
            if title:
                raw = title
    return _clean_title(raw), raw


def _generate_risk_summary(
    score, monthly_rec, feat_rec, contrib_df, llm_result, is_disputed
) -> str:
    """Generate a plain-language summary of why this procurement is flagged."""
    parts = []

    if score is not None and score >= 0.08:
        pct = score * 100
        parts.append(
            f"This procurement has a **{pct:.1f}% risk score**, placing it in the "
            f"**{risk_label(score).lower()} risk** category."
        )
    elif score is not None:
        parts.append(
            f"This procurement has a **{score * 100:.1f}% risk score** (low risk)."
        )

    if is_disputed:
        parts.append(
            "It has **active VAKO dispute(s)** on record."
        )

    # Top positive contributors from ML model
    if contrib_df is not None and len(contrib_df):
        top_pos = contrib_df[contrib_df["contribution"] > 0.05].sort_values(
            "contribution", ascending=False
        )
        if len(top_pos):
            reasons = []
            for _, r in top_pos.head(3).iterrows():
                lbl = FEATURE_EXPLANATIONS.get(r["feature"], (r["feature"], ""))[0]
                reasons.append(lbl.lower())
            parts.append(
                "The main risk drivers are: **"
                + "**, **".join(reasons)
                + "**."
            )

    # LLM-specific findings
    if llm_result and "pass2" in llm_result:
        p2 = llm_result["pass2"]
        issues = p2.get("specific_issues", [])
        high_issues = [i for i in issues if i.get("sustain_probability") in ("high", "medium")]
        if high_issues:
            issue_names = [i.get("issue", "issue") for i in high_issues[:3]]
            parts.append(
                "Document analysis identified: **"
                + "**; **".join(issue_names)
                + "**."
            )
        scenario = p2.get("most_likely_dispute_scenario", "")
        if scenario:
            parts.append(f"*Likely scenario:* {scenario}")
    elif llm_result and llm_result.get("llm_scenario"):
        parts.append(f"*Likely scenario:* {llm_result['llm_scenario']}")

    if not parts:
        return ""
    return "\n\n".join(parts)


def _issue_recommendation(issue_text: str, evidence: str) -> str:
    """Generate a brief fix recommendation based on the issue description."""
    issue_lower = issue_text.lower()
    if "disproportionate" in issue_lower and "turnover" in issue_lower:
        return (
            "Review the turnover requirement against the actual contract value. "
            "RHS \u00a7 38 requires qualification criteria to be proportionate. "
            "Consider reducing the threshold to 1-2x annual contract value, or "
            "allowing consortia to meet the requirement collectively."
        )
    if "disproportionate" in issue_lower and ("qualification" in issue_lower or "experience" in issue_lower):
        return (
            "Ensure qualification requirements are proportionate to the contract scope. "
            "Consider whether similar experience requirements are too narrow "
            "(specific technology, specific client type) and could be broadened "
            "while still ensuring competence."
        )
    if "restrictive" in issue_lower and ("specification" in issue_lower or "technical" in issue_lower):
        return (
            "Review technical specifications for brand-specific or overly narrow requirements. "
            "Use functional specifications where possible (describe what it should do, not what it should be). "
            "If a specific standard is referenced, add 'or equivalent'."
        )
    if "unclear" in issue_lower and "criteria" in issue_lower:
        return (
            "Clarify evaluation methodology: specify exactly how quality criteria will be scored, "
            "what constitutes a high vs low score, and provide the scoring matrix in the tender documents. "
            "Ambiguous criteria are the most common basis for VAKO challenges."
        )
    if "brand" in issue_lower or "vendor" in issue_lower or "lock" in issue_lower:
        return (
            "Remove brand-specific references or add 'or equivalent' language. "
            "Specify requirements in terms of functional performance, not manufacturer. "
            "If a specific brand is genuinely the only option, document the justification."
        )
    if "price" in issue_lower and "only" in issue_lower:
        return (
            "Consider adding quality criteria to the evaluation. For complex services, "
            "price-only evaluation risks selecting providers who undercut on quality. "
            "Even a simple 70/30 price/quality split can improve outcomes."
        )
    # Generic fallback
    return (
        "Review this aspect of the procurement documents for compliance with the "
        "Public Procurement Act (RHS). Consider whether the requirement could be "
        "reworded to be more proportionate, transparent, or competition-friendly."
    )


# ---------------------------------------------------------------------------
# Comparable procurements & benchmarking helpers
# ---------------------------------------------------------------------------

def _find_comparable_procurements(
    features_dict: dict, disputes_data: dict, titles_data: dict,
    sector: str, procedure: str, contract_type: str,
    value: float | None, exclude_rhr: str,
) -> pd.DataFrame:
    """Find similar past procurements and their dispute outcomes."""
    dispute_ids = set(disputes_data.get("disputes", {}).keys())
    rows = []
    for rhr_id, rec in features_dict.items():
        if rhr_id == exclude_rhr:
            continue
        feat = rec.get("features", {})
        # Match sector
        if sector and feat.get(f"sector_{sector}", 0) != 1:
            continue
        # Match procedure type (relaxed: open matches open, neg matches neg)
        if procedure and feat.get(f"proc_{procedure}", 0) != 1:
            continue
        # Match value bracket (within 3x range)
        if value and not feat.get("value_missing", 1):
            rec_value = np.exp(feat.get("log_estimated_value", 0))
            if rec_value < value / 3 or rec_value > value * 3:
                continue
        disputed = rhr_id in dispute_ids
        rec_value = np.exp(feat.get("log_estimated_value", 0)) if not feat.get("value_missing", 1) else None
        title_info = titles_data.get(rhr_id, {})
        rows.append({
            "rhr_id": rhr_id,
            "title": (title_info.get("title", "") or "")[:80],
            "buyer": title_info.get("buyer", rec.get("buyer_name", "")),
            "value": rec_value,
            "disputed": disputed,
            "score": rec.get("stage1_probability", 0),
            "price_weight": feat.get("price_weight", 0),
            "quality_weight": feat.get("quality_weight", 0),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Sort: disputed first, then by score descending
    df = df.sort_values(["disputed", "score"], ascending=[False, False]).head(20)
    return df


def _get_dispute_details(rhr_id: str, disputes_data: dict) -> list[dict]:
    """Get VAKO dispute details for a procurement."""
    proc_disp = disputes_data.get("disputes", {}).get(rhr_id)
    if not proc_disp:
        return []
    disp_list = proc_disp if isinstance(proc_disp, list) else [proc_disp]
    results = []
    for d in disp_list:
        results.append({
            "dispute_id": d.get("disputeId", d.get("id", "")),
            "challenger": d.get("disputerName", d.get("challenger", "")),
            "submitted": d.get("submittedDate", d.get("submitted", "")),
            "status": d.get("statusName", d.get("status", "")),
            "object": d.get("objectName", d.get("object", "")),
            "review_no": d.get("reviewNo", ""),
            "result": d.get("resultName", d.get("result", "")),
        })
    return results


def _generate_action_checklist(
    monthly_rec: dict | None, feat_rec: dict | None,
    llm_result: dict | None, profile: dict | None,
    contrib_df: pd.DataFrame | None,
) -> list[tuple[str, str, str]]:
    """Generate specific actionable recommendations. Returns [(priority, action, rationale)]."""
    actions = []
    procedure = (monthly_rec or {}).get("procedure_type", "")
    contract = (monthly_rec or {}).get("contract_type", "")
    sector = (monthly_rec or {}).get("sector", "")
    value = (monthly_rec or {}).get("estimated_value")

    feat = (feat_rec or {}).get("features", {}) if feat_rec else {}
    pw = feat.get("price_weight", 0)
    qw = feat.get("quality_weight", 0)

    # Value-based recommendations
    if value and value > 5_000_000:
        actions.append((
            "HIGH",
            "Conduct pre-publication legal review of tender documents",
            f"Contracts above \u20ac5M have a 22% dispute rate. At {fmt_eur(value)}, "
            "this procurement is in the highest-risk value bracket. "
            "A legal review before publication can catch issues that would otherwise "
            "lead to VAKO challenges."
        ))

    # Procedure-based recommendations
    if procedure == "neg-w-call":
        actions.append((
            "HIGH",
            "Document justification for negotiated procedure",
            "Negotiated procedures have an 18.2% dispute rate (vs 1.9% for open). "
            "Ensure the choice of procedure is fully documented and that "
            "qualification requirements are proportionate to the contract scope."
        ))
    elif procedure in ("neg-wo-call", "oth-single") and value and value > 200_000:
        actions.append((
            "HIGH",
            "Verify exemption grounds for non-competitive procedure",
            f"Non-competitive procedure on a {fmt_eur(value)} contract requires "
            "documented exemption under RHS \u00a7 28. Ensure the specific legal basis "
            "is cited and the reasoning is recorded."
        ))

    # Evaluation criteria recommendations
    total_w = pw + qw
    is_price_only = (qw == 0 and pw > 0) or (total_w > 0 and qw / total_w < 0.01)
    if is_price_only and contract == "services" and value and value > 200_000:
        actions.append((
            "MEDIUM",
            "Consider adding quality criteria to evaluation",
            "Price-only evaluation for complex services risks selecting providers "
            "who undercut on quality. Even a 70/30 price/quality split with clear "
            "methodology can improve outcomes. VAKO precedents show quality criteria "
            "disputes are easier to defend when methodology is well-documented."
        ))
    elif qw and qw > 0.5:
        actions.append((
            "MEDIUM",
            "Ensure quality criteria have detailed scoring methodology",
            f"Quality weight is {qw:.0%} of evaluation. High quality weights are "
            "the most common basis for VAKO challenges when the scoring methodology "
            "is vague. Specify: what constitutes a high vs low score, provide "
            "examples or a scoring matrix, and define how evaluators will reach consensus."
        ))

    # LLM-specific recommendations
    if llm_result:
        scenario = llm_result.get("llm_scenario", "")
        if "qualification" in scenario.lower() or "experience" in scenario.lower():
            actions.append((
                "MEDIUM",
                "Review qualification requirements for proportionality",
                "The AI analysis identified qualification requirements as a potential "
                "dispute trigger. Ensure turnover requirements are at most 2x annual "
                "contract value (RHS \u00a7 38), and that experience requirements match "
                "the actual contract scope rather than excluding capable newcomers."
            ))
        if "brand" in scenario.lower() or "specific" in scenario.lower():
            actions.append((
                "HIGH",
                "Replace brand-specific requirements with functional specifications",
                "Brand-name restrictions are among the most commonly sustained VAKO "
                "challenges. Replace specific product references with functional "
                "requirements, or add 'or equivalent' language with clear criteria "
                "for evaluating equivalence."
            ))
        if "subjective" in scenario.lower() or "vague" in scenario.lower() or "unclear" in scenario.lower():
            actions.append((
                "MEDIUM",
                "Clarify subjective evaluation criteria",
                "The AI analysis identified potentially vague criteria. "
                "For each quality criterion, document: (1) what is being evaluated, "
                "(2) the scoring scale with descriptions for each level, "
                "(3) how evaluator consensus is reached."
            ))

    # Buyer pattern recommendations
    if profile:
        sbr = profile.get("single_bidder_rate", 0)
        if sbr > 0.3:
            actions.append((
                "LOW",
                "Consider market engagement before publication",
                f"This buyer receives single bids in {sbr:.0%} of procurements. "
                "Consider a prior information notice, market consultation, or "
                "published technical dialogue to increase awareness and competition."
            ))
        if profile.get("vako_disputes", 0) >= 3:
            actions.append((
                "MEDIUM",
                "Review lessons from previous disputes",
                f"This buyer has {profile['vako_disputes']} VAKO disputes on record. "
                "Review past dispute decisions for recurring issues that could be "
                "addressed in this procurement's design."
            ))

    # Sector-specific
    if sector == "IT" and value and value > 1_000_000:
        actions.append((
            "LOW",
            "Check for unintentional vendor lock-in in technical specifications",
            "IT procurements above \u20ac1M with specific technology requirements "
            "frequently face challenges about vendor lock-in. Ensure requirements "
            "describe outcomes, not specific technologies, where possible."
        ))

    # Deduplicate by action text
    seen = set()
    unique = []
    for priority, action, rationale in actions:
        if action not in seen:
            seen.add(action)
            unique.append((priority, action, rationale))
    return unique


def _compute_sector_benchmarks(features_dict: dict, disputes_data: dict, sector: str) -> dict:
    """Compute average metrics for a sector for benchmarking."""
    dispute_ids = set(disputes_data.get("disputes", {}).keys())
    values = []
    dispute_count = 0
    total_count = 0
    price_only_count = 0
    quality_weights = []

    for rhr_id, rec in features_dict.items():
        feat = rec.get("features", {})
        if not feat.get(f"sector_{sector}", 0):
            continue
        total_count += 1
        if rhr_id in dispute_ids:
            dispute_count += 1
        if not feat.get("value_missing", 1):
            values.append(np.exp(feat.get("log_estimated_value", 0)))
        pw = feat.get("price_weight", 0)
        qw = feat.get("quality_weight", 0)
        total_w = pw + qw
        if total_w > 0 and qw / total_w < 0.01:
            price_only_count += 1
        if qw > 0:
            quality_weights.append(qw)

    if total_count == 0:
        return {}
    return {
        "total": total_count,
        "dispute_rate": dispute_count / total_count if total_count else 0,
        "median_value": float(np.median(values)) if values else None,
        "mean_value": float(np.mean(values)) if values else None,
        "price_only_rate": price_only_count / total_count if total_count else 0,
        "avg_quality_weight": float(np.mean(quality_weights)) if quality_weights else 0,
    }


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ProcureSight \u2014 Procurement Risk Intelligence",
    page_icon="\U0001f50d",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar branding */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1e3d 0%, #1a2d5a 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 6px 12px;
        border-radius: 6px;
        transition: background 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15);
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    [data-testid="stMetric"] label {
        color: #64748b !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-weight: 700 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }

    /* Tables */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Headers */
    h1 {
        color: #0f1e3d !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #1e3a5f !important;
    }

    /* Quality score badge */
    .quality-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .quality-excellent { background: #dcfce7; color: #166534; }
    .quality-good { background: #dbeafe; color: #1e40af; }
    .quality-fair { background: #fef3c7; color: #92400e; }
    .quality-poor { background: #fee2e2; color: #991b1b; }

    /* Download button prominence */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(37,99,235,0.3) !important;
    }
    .stDownloadButton > button:hover {
        box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
        transform: translateY(-1px);
    }

    /* Link buttons */
    .stLinkButton > a {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Info/warning/error boxes */
    .stAlert {
        border-radius: 8px !important;
    }

    /* Language toggle radio in sidebar */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio-group"] {
        gap: 8px !important;
        justify-content: center;
    }
    [data-testid="stSidebar"] .stRadio label {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 6px !important;
        padding: 4px 16px !important;
        cursor: pointer;
        transition: all 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: rgba(37,99,235,0.4) !important;
        border-color: rgba(37,99,235,0.6) !important;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.15) !important;
    }
    /* Hide the radio circle indicator */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] div[data-baseweb="radio-markup"] {
        display: none !important;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px !important;
    }

    /* Button styling */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar navigation with session-state driven page switching
# Page keys map to translation keys (never change with language)
_PAGE_KEYS = [
    "page_about",
    "page_risk_monitor",
    "page_compliance",
    "page_integrity",
    "page_historical",
    "page_deep_dive",
]

def _pages() -> list[str]:
    """Return page names in the current language."""
    return [t(k) for k in _PAGE_KEYS]

# Language toggle (initialise before brand header so t() works)
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

# Brand header
st.sidebar.markdown(f"""
<div style="text-align: center; padding: 12px 0 8px 0;">
    <div style="font-size: 1.8rem; font-weight: 800; letter-spacing: 2px; color: #fff;
                text-shadow: 0 2px 8px rgba(0,0,0,0.3);">
        \U0001f50d PROCURESIGHT
    </div>
    <div style="font-size: 0.7rem; color: #94a3b8; letter-spacing: 1px; text-transform: uppercase;">
        {t("sidebar_subtitle")}
    </div>
</div>
""", unsafe_allow_html=True)

# Language toggle
_lang_choice = st.sidebar.radio(
    "Language", ["EN", "ET"],
    index=0 if st.session_state["lang"] == "en" else 1,
    horizontal=True,
    label_visibility="collapsed",
)
if _lang_choice == "EN" and st.session_state["lang"] != "en":
    st.session_state["lang"] = "en"
    st.rerun()
elif _lang_choice == "ET" and st.session_state["lang"] != "et":
    st.session_state["lang"] = "et"
    st.rerun()

PAGES = _pages()

# Allow programmatic navigation via session state
_default_idx = 0
if "navigate_to" in st.session_state:
    target = st.session_state.pop("navigate_to")
    # Match by page key (language-independent) or by current label
    for i, pk in enumerate(_PAGE_KEYS):
        if target == pk or target == t(pk):
            _default_idx = i
            break
    # Also support old English names for backward compat
    _en_names = [TRANSLATIONS[k]["en"] for k in _PAGE_KEYS]
    if target in _en_names:
        _default_idx = _en_names.index(target)

page = st.sidebar.radio(
    "Navigate",
    PAGES,
    index=_default_idx,
    label_visibility="collapsed",
)

st.sidebar.markdown("---")

# CTA in sidebar
st.sidebar.markdown(f"""
<div style="background: rgba(37,99,235,0.15); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
    <div style="font-size: 0.8rem; font-weight: 600; color: #fff; margin-bottom: 4px;">
        {t("sidebar_cta_title")}
    </div>
    <div style="font-size: 0.7rem; color: #94a3b8; line-height: 1.5;">
        {t("sidebar_cta_body")}
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style="font-size: 0.7rem; color: #94a3b8; padding: 0 8px; line-height: 1.8;">
    <div style="display: flex; justify-content: space-between;">
        <span>{t("sidebar_procurements")}</span><span style="font-weight: 600;">57,313</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span>{t("sidebar_disputes")}</span><span style="font-weight: 600;">959</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span>{t("sidebar_data")}</span><span style="font-weight: 600;">2018\u20132026</span>
    </div>
    <div style="display: flex; justify-content: space-between;">
        <span>{t("sidebar_source")}</span><span style="font-weight: 600;">riigihanked.riik.ee</span>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# PAGE 0: About / Landing
# =========================================================================
if page == t("page_about"):

    # Hero section
    st.markdown(f"""
<div style="background: linear-gradient(135deg, #0f1e3d 0%, #1e3a5f 50%, #2563eb 100%);
            padding: 48px 40px; border-radius: 16px; margin-bottom: 24px;">
    <h1 style="color: #fff !important; font-size: 2.6rem; margin: 0; font-weight: 800;">
        \U0001f50d ProcureSight
    </h1>
    <p style="color: #94a3b8; font-size: 1.15rem; margin: 8px 0 24px 0;">
        {t("about_hero_subtitle")}
    </p>
    <p style="color: #e2e8f0; font-size: 1.05rem; max-width: 700px; line-height: 1.7;">
        {t("about_hero_text")}
    </p>
    <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 16px;">
        {t("about_dispute_cost")}: <strong style="color: #fbbf24;">\u20ac5,000\u201350,000+</strong>
    </p>
</div>
    """, unsafe_allow_html=True)

    # Key numbers — user-meaningful
    n1, n2, n3, n4 = st.columns(4)
    n1.metric(t("about_procs_analysed"), "57,313",
              help={"en": "Unique procurements from riigihanked.riik.ee, 2018\u20132026",
                    "et": "Unikaalsed hanked riigihanked.riik.ee-st, 2018\u20132026"}
                    .get(st.session_state.get("lang", "en"), ""))
    n2.metric(t("about_disputes_studied"), "959",
              help={"en": "Real dispute outcomes used to calibrate our risk model",
                    "et": "Tegelikud vaidlustuste tulemused riskimudeli kalibreerimiseks"}
                    .get(st.session_state.get("lang", "en"), ""))
    n3.metric(t("about_risk_signals"), "38",
              help={"en": "Contract value, procedure, criteria, buyer history, integrity checks...",
                    "et": "Lepingu väärtus, menetlus, kriteeriumid, hankija ajalugu, aususe kontrollid..."}
                    .get(st.session_state.get("lang", "en"), ""))
    n4.metric(t("about_coverage"), {"en": "8 years", "et": "8 aastat"}.get(
              st.session_state.get("lang", "en"), "8 years"),
              help={"en": "January 2018 through February 2026",
                    "et": "Jaanuar 2018 kuni veebruar 2026"}
                    .get(st.session_state.get("lang", "en"), ""))

    st.markdown("---")

    # Value proposition: three pillars
    st.markdown(f"### {t('about_what_you_get')}")

    _lang = st.session_state.get("lang", "en")

    p1, p2, p3 = st.columns(3)
    with p1:
        if _lang == "et":
            st.markdown("""
##### Väldi vaidlustusi
Üksainus VAKO vaidlustus maksab **\u20ac5 000\u201350 000+** otseseid kulusid ja kuude pikkust viivitust.
ProcureSight tuvastab konkreetsed mustrid sinu hankes, mis ajalooliselt viivad vaidlustusteni
\u2014 andes sulle aega probleemid **enne avaldamist** parandada.

Üle \u20ac5M hangetel on **22% vaidlustuste määr**. Aga väärtus ei ole ainus tegur \u2014
menetluse valik, kriteeriumide disain ja hankija ajalugu loevad samuti.
            """)
        else:
            st.markdown("""
##### Prevent Disputes
A single VAKO challenge costs **\u20ac5,000\u201350,000+** in direct costs and months of delay.
ProcureSight identifies the specific patterns in your procurement that historically
lead to disputes \u2014 giving you time to fix issues **before publication**.

Procurements over \u20ac5M have a **22% dispute rate**. But it's not just about value \u2014
procedure choice, criteria design, and buyer track record all matter.
            """)

    with p2:
        if _lang == "et":
            st.markdown(f"""
##### Paranda kvaliteeti
Iga hange saab **kvaliteediskoori** (0\u2013100) 5 dimensioonis:
1. {t("qdim_competition")}
2. {t("qdim_criteria")}
3. {t("qdim_strategy")}
4. {t("qdim_transparency")}
5. {t("qdim_integrity")}

Iga dimensioon sisaldab **konkreetseid soovitusi** sinu hanke jaoks.
            """)
        else:
            st.markdown(f"""
##### Improve Quality
Every procurement gets a **Quality Score** (0\u2013100) across 5 dimensions:
1. {t("qdim_competition")}
2. {t("qdim_criteria")}
3. {t("qdim_strategy")}
4. {t("qdim_transparency")}
5. {t("qdim_integrity")}

Each dimension includes **specific recommendations** tailored to your procurement.
            """)

    with p3:
        if _lang == "et":
            st.markdown("""
##### Taga ausus
Automaatne ristkontroll:
- **Poliitiliste annetajate** andmed (ERJK)
- **Äriregister** \u2014 vanus, juhatuse liikmed, omanikud
- **Tegelike kasusaajate** võrgustikud

Tuvastab varjatud kontsentratsiooni (sama omanik, erinevad ettevõtted) ja
poliitilised seosed lepingu võitjatega.
            """)
        else:
            st.markdown("""
##### Ensure Integrity
Automated cross-referencing against:
- **Political donor** records (ERJK)
- **Company registry** — age, board members, ownership
- **Beneficial ownership** networks

Detects hidden concentration (same owner, different companies) and
political connections linked to contract winners.
            """)

    st.markdown("---")

    # How it works — simplified for non-technical audience
    st.markdown(f"### {t('about_how_works')}")
    h1, h2, h3 = st.columns(3)
    with h1:
        if _lang == "et":
            st.markdown("""
**1. Automaatne riskihindamine**

Iga hanget hinnatakse 38 riskisignaali põhjal: lepingu väärtus, menetluse liik,
sektor, hindamiskriteeriumide kaalud, hankija ajalugu ja aususe kontrollid.

Mudel õppis 8 aasta tegelikest tulemustest \u2014 millised hanked vaidlustati
ja millised mitte.
            """)
        else:
            st.markdown("""
**1. Automated Risk Scoring**

Every procurement is scored against 38 risk signals: contract value, procedure type,
sector, evaluation criteria weights, buyer track record, and integrity checks.

The model learned from 8 years of real outcomes \u2014 which procurements got
challenged and which didn't.
            """)
    with h2:
        if _lang == "et":
            st.markdown("""
**2. Dokumentide analüüs**

Kõrge riskiga hangete puhul loeb AI tegelikke hankedokumente
ja võrdleb neid 76 VAKO pretsedendiotsusega.

See tuvastab konkreetsed probleemid \u2014 piiravad spetsifikatsioonid, ebaproportsionaalsed
nõuded, ebamäärased hindamiskriteeriumid.
            """)
        else:
            st.markdown("""
**2. Document Analysis**

For high-risk procurements, AI reads the actual procurement documents
and compares them against 76 VAKO precedent decisions.

It identifies specific issues \u2014 restrictive specifications, disproportionate
requirements, vague evaluation criteria.
            """)
    with h3:
        if _lang == "et":
            st.markdown("""
**3. Tegevuskava**

Iga analüüs sisaldab:
- **Kvaliteediskoor** (0\u2013100) 5 dimensioonis
- **Õiguslikud vastavuse** kontrollid RHS viidetega
- **Aususe märgised** registrite ristkontrollidest
- **Hankija võrdlusanalüüs** sektori keskmistega
- **Prioriseeritud soovitused** konkreetsete parandustega

Laadi alla PDF-raportina või uuri töölaualt.
            """)
        else:
            st.markdown("""
**3. Actionable Report**

Every analysis includes:
- **Quality Score** (0\u2013100) across 5 dimensions
- **Legal compliance** checks with RHS references
- **Integrity flags** from registry cross-references
- **Buyer benchmarking** vs sector averages
- **Prioritised recommendations** with specific fixes

Download as a PDF report or explore in the dashboard.
            """)

    st.markdown("---")

    # Navigation guide — cleaner
    st.markdown(f"### {t('about_explore')}")
    nav_cols = st.columns(5)
    if _lang == "et":
        nav_items = [
            (t("page_risk_monitor"), "Selle kuu hanked riskitaseme järgi"),
            (t("page_compliance"), "Konkreetsed rikkumised ja andmeprobleemid"),
            (t("page_integrity"), "Registrite ristkontrollid ja punased lipud"),
            (t("page_historical"), "8 aasta trendid ja mustrid"),
            (t("page_deep_dive"), "Täisanalüüs PDF-raportiga"),
        ]
    else:
        nav_items = [
            (t("page_risk_monitor"), "This month's procurements ranked by risk"),
            (t("page_compliance"), "Specific rule violations and data issues"),
            (t("page_integrity"), "Registry cross-references and red flags"),
            (t("page_historical"), "8 years of trends and patterns"),
            (t("page_deep_dive"), "Full analysis with PDF report"),
        ]
    for i, (nav_name, nav_desc) in enumerate(nav_items):
        with nav_cols[i]:
            st.markdown(f"**{nav_name}**")
            st.caption(nav_desc)

    st.markdown("---")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(f"#### {t('about_data_sources')}")
        if _lang == "et":
            st.markdown("""
- **Hanketeated** \u2014 riigihanked.riik.ee avatud andmed
- **VAKO vaidlustused** \u2014 959 vaidlustuse tulemust 1097 hanke kohta
- **Äriregister** \u2014 e-Ariregister (366K ettevõtet, juhatuse liikmed, tegelikud kasusaajad)
- **Erakondade rahastamine** \u2014 ERJK annetajate andmed
- **Hankedokumendid** \u2014 avaliku API kaudu
            """)
        else:
            st.markdown("""
- **Procurement notices** \u2014 riigihanked.riik.ee open data
- **VAKO disputes** \u2014 959 dispute outcomes across 1,097 procurements
- **Company registry** \u2014 e-Ariregister (366K companies, board members, beneficial owners)
- **Political financing** \u2014 ERJK donor records
- **Procurement documents** \u2014 via public API
            """)
    with col_d2:
        st.markdown(f"#### {t('about_how_to_read')}")
        if _lang == "et":
            st.markdown("""
- **Riskiskoorid on tõenäosuslikud** \u2014 kõrge skoor tähendab, et sinu hange *sarnaneb*
  ajalooliselt vaidlustatud hangetele, mitte et seda *kindlasti* vaidlustatakse
- **Kvaliteediskoorid näitavad võimalusi** \u2014 kus saad parandada, mitte kus ebaõnnestusid
- **Baasline vaidlustuste määr on ~2%** \u2014 seega 4% skoor tähendab topeltriski
- Kõik andmed pärinevad **avalikest allikatest** \u2014 konfidentsiaalset teavet ei kasutata
            """)
        else:
            st.markdown("""
- **Risk scores are probabilistic** \u2014 a high score means your procurement *resembles*
  historically disputed ones, not that it *will* be disputed
- **Quality scores show opportunities** \u2014 where you can improve, not where you failed
- **Baseline dispute rate is ~2%** \u2014 so a 4% score means double the usual risk
- All data comes from **public sources** \u2014 no confidential information is used
            """)


# =========================================================================
# PAGE 1: Live Risk Monitor
# =========================================================================
elif page == t("page_risk_monitor"):

    months = available_months()
    if not months:
        st.error("No monthly result files found.")
        st.stop()

    selected_month = st.sidebar.selectbox(
        "Month",
        months[::-1],
        format_func=lambda m: f"{m[:4]}-{m[5:]}",
    )

    raw = load_monthly_results(selected_month)
    if raw is None:
        st.error(f"No results for {selected_month}")
        st.stop()

    df = pd.DataFrame(raw["results"])
    disputes = load_disputes()
    dispute_ids = set(disputes.get("disputes", {}).keys())
    df["disputed"] = df["rhr_id"].astype(str).isin(dispute_ids)

    st.title(f"{t('lrm_title')} \u2014 {selected_month[:4]}-{selected_month[5:]}")

    # ---- Metric cards ----
    c1, c2, c3, c4 = st.columns(4)
    elevated_count = (df["stage1_probability"] >= 0.04).sum()
    high_count = (df["stage1_probability"] >= 0.08).sum()
    known = df["disputed"].sum()
    val_series_total = df["estimated_value"].dropna().sum()

    c1.metric(t("lrm_procurements"), f"{len(df):,}")
    c2.metric(t("lrm_elevated_risk"), f"{elevated_count}",
              delta=f"{elevated_count / len(df) * 100:.0f}% {t('lrm_of_total')}" if len(df) else None,
              delta_color="off")
    c3.metric(t("lrm_high_risk"), f"{high_count}",
              delta=f"{high_count / len(df) * 100:.0f}% {t('lrm_of_total')}" if len(df) else None,
              delta_color="inverse" if high_count > 0 else "off")
    c4.metric(t("lrm_known_disputes"), f"{known}")

    # ---- Summary stats row ----
    st.markdown("")
    s1, s2, s3, s4 = st.columns(4)
    val_series = df["estimated_value"].dropna()
    s1.metric(t("lrm_total_value"), fmt_eur(val_series.sum()))
    s2.metric(t("lrm_median_value"), fmt_eur(val_series.median()))
    proc_counts = df["procedure_type"].value_counts()
    most_common_proc = proc_counts.index[0] if len(proc_counts) else "\u2014"
    s3.metric(t("lrm_common_proc"), _procedure_labels().get(most_common_proc, most_common_proc))
    sector_counts = df["sector"].value_counts()
    top_sector = sector_counts.index[0] if len(sector_counts) else "\u2014"
    s4.metric(t("lrm_top_sector"), _sector_labels().get(top_sector, top_sector))

    # ---- Risk tier breakdown ----
    st.markdown("---")
    low_ct = (df["stage1_probability"] < 0.04).sum()
    mod_ct = ((df["stage1_probability"] >= 0.04) & (df["stage1_probability"] < 0.08)).sum()
    elev_ct = ((df["stage1_probability"] >= 0.08) & (df["stage1_probability"] < 0.15)).sum()
    high_ct = (df["stage1_probability"] >= 0.15).sum()

    _tier_data = [
        (t("tier_low"), low_ct, "#22c55e", "#f0fdf4", "<4%"),
        (t("tier_moderate"), mod_ct, "#f59e0b", "#fffbeb", "4\u20138%"),
        (t("tier_elevated"), elev_ct, "#f97316", "#fff7ed", "8\u201315%"),
        (t("tier_high"), high_ct, "#dc2626", "#fef2f2", ">15%"),
    ]
    _tier_html = '<div style="display: flex; gap: 12px; margin-bottom: 16px;">'
    for name, count, color, bg, thresh in _tier_data:
        pct_of_total = count / len(df) * 100 if len(df) else 0
        _tier_html += (
            f'<div style="flex: 1; background: {bg}; border: 1px solid {color}20; '
            f'border-radius: 10px; padding: 12px 16px; border-left: 4px solid {color};">'
            f'<div style="font-size: 0.75rem; color: {color}; text-transform: uppercase; '
            f'font-weight: 600; letter-spacing: 0.5px;">{name}</div>'
            f'<div style="font-size: 1.5rem; font-weight: 700; color: #0f172a;">{count}</div>'
            f'<div style="font-size: 0.7rem; color: #64748b;">{pct_of_total:.0f}% {t("lrm_of_total")} · {thresh}</div>'
            f'</div>'
        )
    _tier_html += '</div>'
    st.markdown(_tier_html, unsafe_allow_html=True)

    # ---- Charts ----
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader(t("lrm_risk_distribution"))
        # Convert to percentage for display
        df["risk_pct_val"] = df["stage1_probability"] * 100
        fig = px.histogram(
            df, x="risk_pct_val", nbins=40,
            color="disputed",
            color_discrete_map={True: "#dc2626", False: "#2563eb"},
            labels={"risk_pct_val": "Risk Score (%)", "disputed": "Disputed"},
            barmode="overlay", opacity=0.7,
        )
        fig.update_layout(
            height=350, margin=dict(t=10, b=30, l=40, r=10),
            legend=dict(orientation="h", yanchor="top", y=0.99, x=0.6),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        )
        # Add reference line at 2% baseline
        fig.add_vline(x=2, line_dash="dot", line_color="#94a3b8",
                       annotation_text="Baseline", annotation_position="top right")
        st.plotly_chart(fig, width="stretch")

    with col_right:
        st.subheader(t("lrm_risk_by_sector"))
        sector_risk = (
            df.groupby("sector")["stage1_probability"]
            .mean().sort_values(ascending=True).reset_index()
        )
        sector_risk["label"] = sector_risk["sector"].map(_sector_labels()).fillna(sector_risk["sector"])
        sector_risk["risk_pct_val"] = sector_risk["stage1_probability"] * 100
        fig = px.bar(
            sector_risk, x="risk_pct_val", y="label", orientation="h",
            labels={"risk_pct_val": "Average Risk (%)", "label": ""},
            color="risk_pct_val",
            color_continuous_scale=["#22c55e", "#f59e0b", "#dc2626"],
        )
        fig.update_layout(
            height=350, margin=dict(t=10, b=30, l=10, r=10),
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_traces(texttemplate="%{x:.2f}%", textposition="outside")
        st.plotly_chart(fig, width="stretch")

    # ---- Key insight callout ----
    _lang = st.session_state.get("lang", "en")
    if high_count > 0 or known > 0:
        _insight_parts = []
        if high_count:
            if _lang == "et":
                _insight_parts.append(
                    f"**{high_count} hange{'t' if high_count > 1 else ''}** sai sel kuul üle 8% skoori, "
                    f"mis viitab tugevale sarnasusele ajalooliselt vaidlustatud hangetega."
                )
            else:
                _insight_parts.append(
                    f"**{high_count} procurement{'s' if high_count > 1 else ''}** scored above 8% "
                    f"this month, indicating strong resemblance to historically disputed cases."
                )
        if known:
            if _lang == "et":
                _insight_parts.append(
                    f"**{known} hanke{'l' if known > 1 else 'l'}** on juba teadaolevad VAKO vaidlustused."
                )
            else:
                _insight_parts.append(
                    f"**{known} procurement{'s' if known > 1 else ''}** already "
                    f"{'have' if known > 1 else 'has'} known VAKO disputes."
                )
        st.info(" ".join(_insight_parts))

    # ---- Risk table with toggle ----
    st.markdown("---")
    st.subheader(t("lrm_by_risk_score"))

    # Filter controls
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    with filter_col1:
        show_count = st.selectbox(t("lrm_show"), ["Top 20", "Top 50", t("lrm_all")], index=0)
    with filter_col2:
        risk_filter = st.selectbox(t("lrm_risk_level"), [
            t("lrm_all"), t("tier_high"), t("tier_elevated"), t("tier_moderate"), t("tier_low")
        ], index=0)
    with filter_col3:
        sector_filter = st.selectbox(t("lrm_sector"), [t("lrm_all")] + sorted(
            [_sector_labels().get(s, s) for s in df["sector"].unique() if s]
        ), index=0)

    titles_data = load_procurement_titles()

    table_df = df.sort_values("stage1_probability", ascending=False).copy()

    # Apply filters
    if risk_filter != t("lrm_all"):
        # Map translated tier names back to English for comparison
        _tier_rev = {t("tier_high"): "High", t("tier_elevated"): "Elevated",
                     t("tier_moderate"): "Moderate", t("tier_low"): "Low"}
        _en_filter = _tier_rev.get(risk_filter, risk_filter)
        table_df["_risk_label"] = table_df["stage1_probability"].apply(risk_label)
        table_df = table_df[table_df["_risk_label"] == _en_filter]
    if sector_filter != t("lrm_all"):
        _rev_sector = {v: k for k, v in _sector_labels().items()}
        sector_key = _rev_sector.get(sector_filter, sector_filter)
        table_df = table_df[table_df["sector"] == sector_key]

    if show_count == "Top 20":
        table_df = table_df.head(20)
    elif show_count == "Top 50":
        table_df = table_df.head(50)

    table_df = table_df.reset_index(drop=True)
    table_df["rank"] = range(1, len(table_df) + 1)
    table_df["value_fmt"] = table_df["estimated_value"].apply(fmt_eur)
    table_df["procedure_label"] = table_df["procedure_type"].map(_procedure_labels()).fillna(table_df["procedure_type"])
    table_df["sector_label"] = table_df["sector"].map(_sector_labels()).fillna(table_df["sector"])
    table_df["risk_pct"] = (table_df["stage1_probability"] * 100).round(2).astype(str) + "%"
    table_df["risk_tier"] = table_df["stage1_probability"].apply(
        lambda p: {"Low": t("tier_low"), "Moderate": t("tier_moderate"),
                    "Elevated": t("tier_elevated"), "High": t("tier_high")}.get(risk_label(p), risk_label(p))
    )
    table_df["dispute_flag"] = table_df["disputed"].map(
        {True: f"\u26a0\ufe0f {t('lrm_disputed')}", False: ""}
    )
    # Enrich buyer name from titles data if missing
    def _enrich_buyer(row):
        if row["buyer_name"]:
            return row["buyer_name"]
        return titles_data.get(str(row["rhr_id"]), {}).get("buyer", "")
    table_df["buyer_display"] = table_df.apply(_enrich_buyer, axis=1)
    # Add short procurement title
    table_df["title"] = table_df["rhr_id"].astype(str).apply(
        lambda rid: _clean_title(titles_data.get(rid, {}).get("title", ""), 60)
    )

    display_cols = {
        "rank": t("col_rank"),
        "risk_pct": t("col_risk"),
        "risk_tier": t("col_level"),
        "title": t("col_procurement"),
        "buyer_display": t("col_buyer"),
        "sector_label": t("col_sector"),
        "procedure_label": t("col_procedure"),
        "value_fmt": t("col_value"),
        "dispute_flag": t("col_status"),
    }
    st.markdown(
        f'<p style="color: #2563eb; font-size: 0.85rem; margin-bottom: 4px;">'
        f'\u261d {t("lrm_click_row")}</p>',
        unsafe_allow_html=True,
    )
    event = st.dataframe(
        table_df[list(display_cols.keys())].rename(columns=display_cols),
        width="stretch",
        hide_index=True,
        height=min(len(table_df) * 35 + 38, 800),
        on_select="rerun",
        selection_mode="single-row",
    )

    # Navigate to deep dive on row click
    if event and event.selection and event.selection.rows:
        clicked_idx = event.selection.rows[0]
        if clicked_idx < len(table_df):
            clicked_rhr = str(table_df.iloc[clicked_idx]["rhr_id"])
            st.session_state["navigate_to"] = "page_deep_dive"
            st.session_state["deep_dive_rhr"] = clicked_rhr
            st.rerun()

    # ---- Combined / LLM results ----
    combined = load_combined_results(selected_month)
    if combined and combined.get("results"):
        with st.expander(f"Stage 2 \u2014 LLM Analysis ({combined['stage2_count']} procurements)", expanded=False):
            cdf = pd.DataFrame(combined["results"])
            cdf["combined_pct"] = (cdf["combined_score"] * 100).round(1).astype(str) + "%"
            cdf["stage1_pct"] = (cdf["stage1_probability"] * 100).round(2).astype(str) + "%"
            cdf["llm_display"] = cdf["llm_score"].astype(str) + "/10"
            show_cols = {
                "rhr_id": "RHR ID",
                "buyer_name": "Buyer",
                "stage1_pct": "Stage 1",
                "llm_display": "LLM Score",
                "llm_confidence": "Confidence",
                "combined_pct": "Combined",
                "llm_scenario": "Scenario",
            }
            st.dataframe(
                cdf[list(show_cols.keys())].rename(columns=show_cols),
                width="stretch",
                hide_index=True,
            )


# =========================================================================
# PAGE 1.5: Compliance Scan (Clear Errors)
# =========================================================================
elif page == t("page_compliance"):

    months = available_months()
    if not months:
        st.error("No monthly result files found.")
        st.stop()

    selected_month = st.sidebar.selectbox(
        "Month",
        months[::-1],
        format_func=lambda m: f"{m[:4]}-{m[5:]}",
    )

    raw = load_monthly_results(selected_month)
    if raw is None:
        st.error(f"No results for {selected_month}")
        st.stop()

    df = pd.DataFrame(raw["results"])
    buyer_profiles = load_buyer_profiles()
    titles_data = load_procurement_titles()

    st.title(f"{t('comp_title')} \u2014 {selected_month[:4]}-{selected_month[5:]}")
    st.markdown(t("comp_intro"))

    # Rule explanations
    with st.expander("What does each rule check?", expanded=False):
        for rule_id, (rtitle, rexpl, severity) in COMPLIANCE_RULES.items():
            sev_icon = {"high": "\U0001f534", "medium": "\U0001f7e0", "low": "\u26aa"}.get(severity, "")
            st.markdown(f"**{sev_icon} {rtitle}**")
            st.markdown(f"{rexpl}")
            st.markdown("")

    errors_df = detect_clear_errors(df, buyer_profiles, titles_data)

    if errors_df.empty:
        st.success("No clear compliance issues detected in this month's procurements.")
        st.stop()

    # Severity breakdown
    _severity_counts = {"high": 0, "medium": 0, "low": 0}
    for _, row in errors_df.iterrows():
        sev = COMPLIANCE_RULES.get(row.get("rule_id", ""), ("", "", "medium"))[2]
        _severity_counts[sev] = _severity_counts.get(sev, 0) + 1

    st.subheader(f"{len(errors_df)} issues across {errors_df['rhr_id'].nunique()} procurements")

    _sev_html = '<div style="display: flex; gap: 12px; margin-bottom: 16px;">'
    _sev_items = [
        (t("comp_critical"), _severity_counts["high"], "#dc2626", "#fef2f2"),
        (t("comp_warning"), _severity_counts["medium"], "#f59e0b", "#fffbeb"),
        (t("comp_info"), _severity_counts["low"], "#6b7280", "#f9fafb"),
    ]
    for sev_name, sev_ct, sev_color, sev_bg in _sev_items:
        _sev_html += (
            f'<div style="flex: 1; background: {sev_bg}; border-left: 4px solid {sev_color}; '
            f'border-radius: 8px; padding: 12px 16px;">'
            f'<div style="font-size: 0.75rem; color: {sev_color}; font-weight: 600; '
            f'text-transform: uppercase;">{sev_name}</div>'
            f'<div style="font-size: 1.5rem; font-weight: 700; color: #0f172a;">{sev_ct}</div>'
            f'</div>'
        )
    _sev_html += '</div>'
    st.markdown(_sev_html, unsafe_allow_html=True)

    # Issue type counts
    issue_counts = errors_df["issue_title"].value_counts()
    cols = st.columns(min(len(issue_counts), 4))
    for i, (title, count) in enumerate(issue_counts.items()):
        cols[i % len(cols)].metric(title, count)

    st.markdown("---")

    # Filter by issue type
    issue_filter = st.multiselect(
        "Filter by issue type",
        options=issue_counts.index.tolist(),
        default=issue_counts.index.tolist(),
    )
    filtered = errors_df[errors_df["issue_title"].isin(issue_filter)]

    # Show each issue as an expandable card
    for _, row in filtered.iterrows():
        rule_id = row.get("rule_id", "")
        severity = COMPLIANCE_RULES.get(rule_id, ("", "", "medium"))[2]
        sev_icon = {"high": "\U0001f534", "medium": "\U0001f7e0", "low": "\u26aa"}.get(severity, "\U0001f534")
        buyer = row["buyer_name"] or "Unknown buyer"
        # Include title if available
        proc_title = titles_data.get(str(row["rhr_id"]), {}).get("title", "")
        label_suffix = f" \u2014 {proc_title[:50]}" if proc_title else ""
        with st.expander(
            f'{sev_icon} {row["issue_title"]} \u2014 {buyer} ({row["rhr_id"]}){label_suffix}',
            expanded=False,
        ):
            ic1, ic2, ic3 = st.columns(3)
            ic1.markdown(f"**{t('col_sector')}:** {_sector_labels().get(row['sector'], row['sector'])}")
            ic2.markdown(f"**{t('col_procedure')}:** {_procedure_labels().get(row['procedure_type'], row['procedure_type'])}")
            ic3.markdown(f"**{t('col_value')}:** {fmt_eur(row['estimated_value'])}")

            if proc_title:
                st.markdown(f"**Procurement:** {proc_title}")

            st.markdown(f"**Finding:** {row['issue_detail']}")

            # Show rule explanation
            rule_expl = COMPLIANCE_RULES.get(rule_id, ("", "", ""))[1]
            if rule_expl:
                st.caption(f"Rule: {rule_expl}")

            _comp_link_col1, _comp_link_col2 = st.columns(2)
            with _comp_link_col1:
                if st.button(t("dd_title"), key=f"dd_{row['rhr_id']}_{row['rule_id']}"):
                    st.session_state["navigate_to"] = "page_deep_dive"
                    st.session_state["deep_dive_rhr"] = str(row["rhr_id"])
                    st.rerun()
            with _comp_link_col2:
                st.markdown(
                    f"[View on riigihanked.riik.ee \u2192](https://riigihanked.riik.ee/rhr-web/#/procurement/{row['rhr_id']}/procurement-passport)"
                )


# =========================================================================
# PAGE 2: Integrity Analysis
# =========================================================================
elif page == t("page_integrity"):

    st.title(t("int_title"))
    st.markdown(t("int_intro"))

    integrity = load_integrity_lookups()
    phase2 = load_phase2_results()
    gap = load_gap_analysis()
    enriched = load_enriched_procurements()
    features_dict = load_features()
    titles_data = load_procurement_titles()
    disputes_data = load_disputes()
    dispute_ids = set(disputes_data.get("disputes", {}).keys())

    # --- Summary metrics ---
    n_donor = len(integrity.get("donor_linked", {}))
    n_concentration = len(integrity.get("hidden_concentration", {}))
    n_threshold = len(integrity.get("threshold_proximity", {}))
    n_young = sum(1 for a in integrity.get("winner_age_years", {}).values() if a < 2)
    n_cpv_anomaly = sum(1 for z in integrity.get("cpv_price_zscore", {}).values() if z > 2.0)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric(t("int_political_donors"), f"{n_donor:,}")
    mc2.metric(t("int_hidden_ownership"), f"{n_concentration:,}")
    mc3.metric(t("int_near_threshold"), f"{n_threshold:,}")
    mc4.metric(t("int_young_winners"), f"{n_young:,}")
    mc5.metric(t("int_price_anomalies"), f"{n_cpv_anomaly:,}")

    # Compute base dispute rate for comparisons
    all_disputed = sum(1 for r in features_dict.values() if r.get("has_dispute"))
    base_rate = all_disputed / len(features_dict) * 100 if features_dict else 0

    st.markdown("---")

    # === Tab layout for sub-analyses ===
    tab_donor, tab_ownership, tab_age, tab_threshold, tab_cpv = st.tabs([
        t("int_political_donors"),
        t("int_hidden_ownership"),
        {"en": "Company Age", "et": "Ettevõtte vanus"}.get(st.session_state.get("lang", "en"), "Company Age"),
        {"en": "Threshold Proximity", "et": "Piirmäära lähedus"}.get(st.session_state.get("lang", "en"), "Threshold Proximity"),
        {"en": "CPV Price Anomalies", "et": "CPV hinna anomaaliad"}.get(st.session_state.get("lang", "en"), "CPV Price Anomalies"),
    ])

    # --- Tab 1: Political Donors ---
    with tab_donor:
        st.subheader("Political Donor \u2192 Contract Winners")
        st.markdown(
            "Board members of winning companies cross-referenced against ERJK "
            "(party financing supervisory commission) records. Only **material donors** "
            "(\u22655,000 EUR total) are flagged \u2014 small party membership fees are filtered out."
        )

        donor_ids = integrity.get("donor_linked", {})
        if donor_ids:
            # Dispute rate comparison
            donor_disputed = sum(1 for rid in donor_ids if rid in dispute_ids)
            donor_rate = donor_disputed / len(donor_ids) * 100 if donor_ids else 0

            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Donor-Linked Contracts", f"{len(donor_ids):,}")
            dc2.metric("Dispute Rate", f"{donor_rate:.1f}%",
                        delta=f"{donor_rate - base_rate:+.1f}% vs baseline {base_rate:.1f}%",
                        delta_color="inverse")
            dc3.metric("Lift", f"{donor_rate / base_rate:.1f}x" if base_rate > 0 else "N/A")

            # Top donor-linked procurements table
            phase2_tests = {t["test"]: t for t in phase2.get("tests", [])}
            donor_test = phase2_tests.get("political_donations", {})
            top_companies = donor_test.get("top_companies", [])

            if top_companies:
                st.markdown("#### Top Companies with Donor-Linked Board Members")
                comp_rows = []
                for c in top_companies[:20]:
                    comp_rows.append({
                        "Company": c.get("name", ""),
                        "Reg Code": c.get("code", ""),
                        "Contracts": c.get("count", 0),
                        "Total Value": fmt_eur(c.get("value", 0)),
                        "Donor(s)": ", ".join(c.get("donors", []))[:60],
                    })
                st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

            # Show donor-linked procurements with highest risk scores
            st.markdown("#### Highest-Risk Donor-Linked Procurements")
            donor_procs = []
            for rid in donor_ids:
                feat_rec = features_dict.get(rid)
                if not feat_rec:
                    continue
                enr = enriched.get(rid, {})
                title_info = titles_data.get(rid, {})
                donor_procs.append({
                    "rhr_id": rid,
                    "Buyer": feat_rec.get("buyer_name", "") or title_info.get("buyer", ""),
                    "Title": (title_info.get("title", "") or "")[:60],
                    "Value": enr.get("estimated_value"),
                    "Winner": enr.get("winner_name", ""),
                    "Disputed": rid in dispute_ids,
                    "score": feat_rec.get("features", {}).get("log_estimated_value", -1),
                })
            if donor_procs:
                dp_df = pd.DataFrame(donor_procs).sort_values("score", ascending=False).head(20)
                dp_df["Value"] = dp_df["Value"].apply(fmt_eur)
                dp_df["Status"] = dp_df["Disputed"].map({True: "DISPUTED", False: ""})
                st.dataframe(
                    dp_df[["rhr_id", "Buyer", "Title", "Value", "Winner", "Status"]],
                    width="stretch", hide_index=True,
                )
        else:
            st.info("No political donor linkage data available. Run compute_integrity_features.py.")

    # --- Tab 2: Ownership Networks ---
    with tab_ownership:
        st.subheader("Hidden Ownership Concentration")
        st.markdown(
            "Identifies cases where **different companies winning contracts from the same buyer** "
            "share a common beneficial owner. This can indicate undisclosed related-party "
            "transactions or coordinated bidding."
        )

        conc_ids = integrity.get("hidden_concentration", {})
        if conc_ids:
            conc_disputed = sum(1 for rid in conc_ids if rid in dispute_ids)
            conc_rate = conc_disputed / len(conc_ids) * 100 if conc_ids else 0

            oc1, oc2, oc3 = st.columns(3)
            oc1.metric("Flagged Procurements", f"{len(conc_ids):,}")
            oc2.metric("Dispute Rate", f"{conc_rate:.1f}%",
                        delta=f"{conc_rate - base_rate:+.1f}% vs baseline",
                        delta_color="inverse")
            oc3.metric("Lift", f"{conc_rate / base_rate:.1f}x" if base_rate > 0 else "N/A")

            # Top ownership overlaps from phase2
            phase2_tests = {t["test"]: t for t in phase2.get("tests", [])}
            ownership_test = phase2_tests.get("ownership_networks", {})
            top_overlaps = ownership_test.get("top_overlaps", [])

            if top_overlaps:
                st.markdown("#### Top Same-Owner, Different-Company Cases at Same Buyer")
                ov_rows = []
                for ov in top_overlaps[:20]:
                    companies = ov.get("companies_winning", [])
                    ov_rows.append({
                        "Owner": ov.get("owner", ""),
                        "Buyer": ov.get("buyer", ""),
                        "Companies": ", ".join(companies[:3]) + ("..." if len(companies) > 3 else ""),
                        "Contracts": ov.get("contract_count", 0),
                        "Total Value": fmt_eur(ov.get("total_value", 0)),
                    })
                st.dataframe(pd.DataFrame(ov_rows), width="stretch", hide_index=True)

            # Most connected persons
            top_connected = ownership_test.get("top_connected", [])
            if top_connected:
                st.markdown("#### Most Connected Individuals (Multiple Winning Companies)")
                conn_rows = []
                for tc in top_connected[:15]:
                    winner_cos = tc.get("winner_companies", [])
                    conn_rows.append({
                        "Person": tc.get("person_name", ""),
                        "Winner Companies": tc.get("winner_company_count", 0),
                        "Total Companies": tc.get("total_company_count", 0),
                        "Company Names": ", ".join(
                            c.get("name", "") for c in winner_cos[:3]
                        ) + ("..." if len(winner_cos) > 3 else ""),
                    })
                st.dataframe(pd.DataFrame(conn_rows), width="stretch", hide_index=True)
        else:
            st.info("No ownership network data available. Run compute_integrity_features.py.")

    # --- Tab 3: Company Age ---
    with tab_age:
        st.subheader("Winner Company Age at Contract Award")
        st.markdown(
            "Cross-references winning companies against the e-Ariregister to check "
            "how old each company was when it won the contract. Very young companies "
            "winning large contracts may indicate shell entities."
        )

        ages = integrity.get("winner_age_years", {})
        if ages:
            age_vals = list(ages.values())
            under_1 = sum(1 for a in age_vals if a < 1)
            under_2 = sum(1 for a in age_vals if a < 2)
            under_5 = sum(1 for a in age_vals if a < 5)

            ac1, ac2, ac3, ac4 = st.columns(4)
            ac1.metric("Companies Matched", f"{len(ages):,}")
            ac2.metric("Under 1 Year", f"{under_1:,}")
            ac3.metric("Under 2 Years", f"{under_2:,}")
            ac4.metric("Under 5 Years", f"{under_5:,}")

            # Age distribution chart
            age_bins = [0, 1, 2, 5, 10, 20, 50, 100]
            bin_labels = ["<1yr", "1-2yr", "2-5yr", "5-10yr", "10-20yr", "20-50yr", "50+yr"]
            bin_counts = [0] * len(bin_labels)
            for a in age_vals:
                for i in range(len(age_bins) - 1):
                    if age_bins[i] <= a < age_bins[i + 1]:
                        bin_counts[i] += 1
                        break

            fig = go.Figure(go.Bar(
                x=bin_labels, y=bin_counts,
                marker_color=["#d32f2f", "#f57c00", "#fbc02d", "#66bb6a", "#388e3c", "#2e7d32", "#1b5e20"],
            ))
            fig.update_layout(
                title="Winner Company Age Distribution",
                xaxis_title="Company Age at Contract Award",
                yaxis_title="Number of Contracts",
                height=350,
            )
            st.plotly_chart(fig, width="stretch")

            # Young companies with high-value contracts
            phase2_tests = {t["test"]: t for t in phase2.get("tests", [])}
            age_test = phase2_tests.get("company_age", {})
            high_val_young = age_test.get("high_value_young", [])

            if high_val_young:
                st.markdown("#### Youngest Companies with Highest-Value Contracts")
                yc_rows = []
                for yc in high_val_young[:20]:
                    yc_rows.append({
                        "rhr_id": yc.get("rhr_id", ""),
                        "Winner": yc.get("winner_name", ""),
                        "Buyer": yc.get("buyer_name", ""),
                        "Value": fmt_eur(yc.get("estimated_value")),
                        "Age (years)": f"{yc.get('age_years', 0):.1f}",
                        "Sector": _sector_labels().get(yc.get("sector", ""), yc.get("sector", "")),
                    })
                st.dataframe(pd.DataFrame(yc_rows), width="stretch", hide_index=True)
        else:
            st.info("No company age data available. Run compute_integrity_features.py.")

    # --- Tab 4: Threshold Proximity ---
    with tab_threshold:
        st.subheader("EU Threshold Proximity")
        st.markdown(
            "Procurements valued at **90-99%** of an EU procurement threshold "
            "(\u20ac143K services, \u20ac443K utilities, \u20ac5.5M works). This pattern can indicate "
            "deliberate threshold avoidance to escape stricter EU-level procedures."
        )

        thresh_data = integrity.get("threshold_proximity", {})
        if thresh_data:
            tc1, tc2 = st.columns(2)
            tc1.metric("Near-Threshold Contracts", f"{len(thresh_data):,}")
            thresh_disputed = sum(1 for rid in thresh_data if rid in dispute_ids)
            thresh_rate = thresh_disputed / len(thresh_data) * 100 if thresh_data else 0
            tc2.metric("Dispute Rate", f"{thresh_rate:.1f}%",
                        delta=f"{thresh_rate - base_rate:+.1f}% vs baseline",
                        delta_color="inverse")

            # Distribution by threshold
            thresholds = {"143K (services)": 143_000, "443K (utilities)": 443_000, "5.5M (works)": 5_538_000}
            thresh_counts = {k: 0 for k in thresholds}
            for rid, pct in thresh_data.items():
                enr = enriched.get(rid, {})
                val = enr.get("estimated_value", 0) or 0
                if val > 0:
                    for label, t_val in thresholds.items():
                        if 0.90 <= val / t_val < 1.0:
                            thresh_counts[label] += 1
                            break

            fig = go.Figure(go.Bar(
                x=list(thresh_counts.keys()),
                y=list(thresh_counts.values()),
                marker_color=["#f57c00", "#fbc02d", "#d32f2f"],
            ))
            fig.update_layout(
                title="Contracts Near Each EU Threshold",
                xaxis_title="EU Threshold",
                yaxis_title="Count (90-99% of threshold)",
                height=300,
            )
            st.plotly_chart(fig, width="stretch")

            # Table of closest-to-threshold
            st.markdown("#### Contracts Closest to EU Thresholds")
            thresh_procs = []
            for rid, pct in sorted(thresh_data.items(), key=lambda x: -x[1])[:20]:
                enr = enriched.get(rid, {})
                title_info = titles_data.get(rid, {})
                thresh_procs.append({
                    "rhr_id": rid,
                    "Buyer": enr.get("buyer_name", "") or title_info.get("buyer", ""),
                    "Value": fmt_eur(enr.get("estimated_value")),
                    "% of Threshold": f"{pct * 100:.1f}%",
                    "Status": "DISPUTED" if rid in dispute_ids else "",
                })
            st.dataframe(pd.DataFrame(thresh_procs), width="stretch", hide_index=True)
        else:
            st.info("No threshold proximity data available. Run compute_integrity_features.py.")

    # --- Tab 5: CPV Price Anomalies ---
    with tab_cpv:
        st.subheader("CPV-4 Price Benchmarking")
        st.markdown(
            "Each contract value is compared to the **median value** for its CPV-4 category "
            "(4-digit CPV code = sector group). Contracts with a z-score above 2.0 are "
            "significantly more expensive than typical for their category."
        )

        zscores = integrity.get("cpv_price_zscore", {})
        if zscores:
            z_vals = list(zscores.values())
            above_2 = sum(1 for z in z_vals if z > 2.0)
            above_3 = sum(1 for z in z_vals if z > 3.0)

            zc1, zc2, zc3, zc4 = st.columns(4)
            zc1.metric("Benchmarked", f"{len(zscores):,}")
            zc2.metric("Above 2\u03c3", f"{above_2:,}")
            zc3.metric("Above 3\u03c3", f"{above_3:,}")
            anom_disputed = sum(1 for rid, z in zscores.items() if z > 2.0 and rid in dispute_ids)
            anom_rate = anom_disputed / above_2 * 100 if above_2 else 0
            zc4.metric("Anomaly Dispute Rate", f"{anom_rate:.1f}%")

            # Z-score distribution
            z_bins = list(range(-5, 8))
            z_bin_labels = [f"{z}" for z in z_bins[:-1]]
            z_bin_counts = [0] * (len(z_bins) - 1)
            for z in z_vals:
                z_clamped = max(-5, min(6, int(z)))
                idx = z_clamped + 5
                if 0 <= idx < len(z_bin_counts):
                    z_bin_counts[idx] += 1

            colors = []
            for z in z_bins[:-1]:
                if z >= 3:
                    colors.append("#d32f2f")
                elif z >= 2:
                    colors.append("#f57c00")
                elif z >= 1:
                    colors.append("#fbc02d")
                else:
                    colors.append("#66bb6a")

            fig = go.Figure(go.Bar(x=z_bin_labels, y=z_bin_counts, marker_color=colors))
            fig.update_layout(
                title="CPV-4 Price Z-Score Distribution",
                xaxis_title="Z-Score (standard deviations from CPV-4 median)",
                yaxis_title="Number of Contracts",
                height=350,
            )
            # Mark the 2σ threshold bar (index 7 = z-score 2 in our -5..6 range)
            fig.add_shape(
                type="line", x0=6.5, x1=6.5, y0=0, y1=max(z_bin_counts) * 1.1,
                line=dict(color="red", width=2, dash="dash"),
            )
            fig.add_annotation(x=6.5, y=max(z_bin_counts) * 1.05,
                               text="2\u03c3 threshold", showarrow=False,
                               font=dict(color="red", size=11))
            st.plotly_chart(fig, width="stretch")

            # Top anomalies
            st.markdown("#### Most Anomalous Contract Values")
            anom_procs = []
            for rid, z in sorted(zscores.items(), key=lambda x: -x[1])[:20]:
                enr = enriched.get(rid, {})
                title_info = titles_data.get(rid, {})
                anom_procs.append({
                    "rhr_id": rid,
                    "Buyer": enr.get("buyer_name", "") or title_info.get("buyer", ""),
                    "CPV": enr.get("cpv_code", "")[:4],
                    "Value": fmt_eur(enr.get("estimated_value")),
                    "Z-Score": f"{z:.1f}\u03c3",
                    "Status": "DISPUTED" if rid in dispute_ids else "",
                })
            st.dataframe(pd.DataFrame(anom_procs), width="stretch", hide_index=True)
        else:
            st.info("No CPV price benchmark data available. Run compute_integrity_features.py.")


# =========================================================================
# PAGE 3: Historical Analysis
# =========================================================================
elif page == t("page_historical"):

    st.title(t("hist_title"))
    st.markdown(t("hist_subtitle"))

    scores_df = load_all_scores()
    disputes = load_disputes()
    dispute_ids = set(disputes.get("disputes", {}).keys())

    scores_df["date"] = pd.to_datetime(
        scores_df["source_month"].str.replace("_", "-") + "-01"
    )
    scores_df["year"] = scores_df["date"].dt.year
    scores_df["disputed"] = scores_df["rhr_id"].astype(str).isin(dispute_ids)

    deduped = scores_df.drop_duplicates(subset="rhr_id", keep="first")

    # Key findings callout
    _lang = st.session_state.get("lang", "en")
    if _lang == "et":
        _kf_body = (
            'Lepingu väärtus on vaidlustuste <strong>#1 ennustaja</strong>: '
            'üle \u20ac5M hangetel on <strong>22% vaidlustuste määr</strong> vs peaaegu null alla \u20ac1M. '
            'Läbirääkimistega menetlustel on <strong>18,2% vaidlustuste määr</strong> (vs 1,9% avatud). '
            'Energeetikasektor juhib <strong>7,7%</strong>-ga, ehitus vaid 1,5%.'
        )
    else:
        _kf_body = (
            'Contract value is the <strong>#1 predictor</strong> of disputes: '
            'procurements over \u20ac5M have a <strong>22% dispute rate</strong> vs near-zero under \u20ac1M. '
            'Negotiated procedures have an <strong>18.2% dispute rate</strong> (vs 1.9% for open). '
            'Energy sector leads at <strong>7.7%</strong>, while construction is only 1.5%.'
        )
    st.markdown(
        '<div style="background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 10px; '
        'padding: 16px 20px; margin-bottom: 20px;">'
        f'<div style="font-weight: 700; color: #1e40af; margin-bottom: 6px;">{t("hist_key_findings")}</div>'
        f'<div style="color: #1e3a5f; font-size: 0.9rem; line-height: 1.6;">{_kf_body}</div></div>',
        unsafe_allow_html=True,
    )

    # ---- Row 1: Monthly volume ----
    st.subheader(t("hist_monthly_volume"))
    monthly = scores_df.groupby("source_month").agg(
        total=("rhr_id", "count"),
        disputes=("disputed", "sum"),
    ).reset_index()
    monthly["date"] = pd.to_datetime(monthly["source_month"].str.replace("_", "-") + "-01")
    monthly = monthly.sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["date"], y=monthly["total"],
        name="Procurements", mode="lines+markers",
        line=dict(color="#2563eb", width=2), marker=dict(size=4),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.05)",
    ))
    fig.add_trace(go.Bar(
        x=monthly["date"], y=monthly["disputes"],
        name="Disputes", marker_color="#dc2626", opacity=0.7,
        yaxis="y2",
    ))
    fig.update_layout(
        height=380, margin=dict(t=10, b=30),
        legend=dict(orientation="h", yanchor="top", y=1.12, x=0),
        yaxis=dict(title="Procurements", showgrid=True, gridcolor="#f1f5f9"),
        yaxis2=dict(title="Disputes", overlaying="y", side="right",
                    showgrid=False, rangemode="tozero"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")

    # ---- Row 2: Value brackets & Sector ----
    features_dict = load_features()
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader(t("hist_by_value"))
        feat_rows = []
        for rhr_id, rec in features_dict.items():
            feat = rec.get("features", {})
            feat_rows.append({
                "rhr_id": rhr_id,
                "log_value": feat.get("log_estimated_value", -1),
                "value_missing": feat.get("value_missing", 1),
                "has_dispute": rec.get("has_dispute", False),
            })
        vdf = pd.DataFrame(feat_rows)
        vdf = vdf[vdf["value_missing"] == 0].copy()
        vdf["value_eur"] = np.exp(vdf["log_value"])

        bins = [0, 50_000, 200_000, 1_000_000, 5_000_000, 20_000_000, float("inf")]
        bin_labels = ["<\u20ac50K", "\u20ac50K-200K", "\u20ac200K-1M", "\u20ac1M-5M", "\u20ac5M-20M", ">\u20ac20M"]
        vdf["bracket"] = pd.cut(vdf["value_eur"], bins=bins, labels=bin_labels)
        bracket_stats = vdf.groupby("bracket", observed=True).agg(
            total=("rhr_id", "count"),
            disputes=("has_dispute", "sum"),
        ).reset_index()
        bracket_stats["rate"] = (bracket_stats["disputes"] / bracket_stats["total"] * 100).round(1)

        fig = px.bar(
            bracket_stats, x="bracket", y="rate",
            labels={"bracket": "Contract Value", "rate": "Dispute Rate (%)"},
            color="rate",
            color_continuous_scale=["#388e3c", "#fbc02d", "#d32f2f"],
            text="rate",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            height=350, margin=dict(t=10, b=30),
            coloraxis_showscale=False, showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    with col_r:
        st.subheader(t("hist_by_sector"))
        sector_cols = [c for c in list(features_dict.values())[0].get("features", {}).keys()
                       if c.startswith("sector_")]
        sec_rows = []
        for rhr_id, rec in features_dict.items():
            feat = rec.get("features", {})
            for sc in sector_cols:
                if feat.get(sc, 0) == 1:
                    sec_rows.append({
                        "sector": sc.replace("sector_", ""),
                        "has_dispute": rec.get("has_dispute", False),
                    })
                    break
        sdf = pd.DataFrame(sec_rows)
        if len(sdf):
            sec_stats = sdf.groupby("sector").agg(
                total=("has_dispute", "count"),
                disputes=("has_dispute", "sum"),
            ).reset_index()
            sec_stats["rate"] = (sec_stats["disputes"] / sec_stats["total"] * 100).round(1)
            sec_stats["label"] = sec_stats["sector"].map(_sector_labels()).fillna(sec_stats["sector"])
            sec_stats = sec_stats.sort_values("rate", ascending=True)

            fig = px.bar(
                sec_stats, x="rate", y="label", orientation="h",
                labels={"rate": "Dispute Rate (%)", "label": ""},
                color="rate",
                color_continuous_scale=["#388e3c", "#fbc02d", "#d32f2f"],
                text="rate",
            )
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(
                height=350, margin=dict(t=10, b=30, l=10),
                coloraxis_showscale=False, showlegend=False,
            )
            st.plotly_chart(fig, width="stretch")

    # ---- Row 3: Procedure type ----
    st.subheader(t("hist_by_procedure"))
    proc_cols = [c for c in list(features_dict.values())[0].get("features", {}).keys()
                 if c.startswith("proc_")]
    proc_rows = []
    for rhr_id, rec in features_dict.items():
        feat = rec.get("features", {})
        for pc in proc_cols:
            if feat.get(pc, 0) == 1:
                proc_rows.append({
                    "procedure": pc.replace("proc_", ""),
                    "has_dispute": rec.get("has_dispute", False),
                })
                break
    pdf_proc = pd.DataFrame(proc_rows)
    if len(pdf_proc):
        proc_stats = pdf_proc.groupby("procedure").agg(
            total=("has_dispute", "count"),
            disputes=("has_dispute", "sum"),
        ).reset_index()
        proc_stats["rate"] = (proc_stats["disputes"] / proc_stats["total"] * 100).round(1)
        proc_stats["label"] = proc_stats["procedure"].map(_procedure_labels()).fillna(proc_stats["procedure"])
        proc_stats = proc_stats.sort_values("rate", ascending=True)

        fig = px.bar(
            proc_stats, x="rate", y="label", orientation="h",
            labels={"rate": "Dispute Rate (%)", "label": ""},
            color="rate",
            color_continuous_scale=["#388e3c", "#fbc02d", "#d32f2f"],
            text="rate",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(
            height=300, margin=dict(t=10, b=30, l=10),
            coloraxis_showscale=False, showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    # ---- Row 4: Buyer risk rankings ----
    st.subheader(t("hist_top_buyers"))
    profiles = load_buyer_profiles()
    bp_df = pd.DataFrame(profiles.values())
    bp_df = bp_df[bp_df["procurement_count"] >= 5].copy()
    bp_top = bp_df.nlargest(20, "risk_score").copy()
    bp_top["price_only_pct"] = (bp_top["price_only_rate"] * 100).round(1).astype(str) + "%"
    bp_top["single_bid_pct"] = (bp_top["single_bidder_rate"] * 100).round(1).astype(str) + "%"
    bp_top["risk_pct"] = (bp_top["risk_score"] * 100).round(1).astype(str) + "%"
    bp_top["flags_str"] = bp_top["risk_flags"].apply(lambda x: ", ".join(x) if isinstance(x, list) else "")

    buyer_cols = {
        "buyer_name": "Buyer",
        "procurement_count": "Procs",
        "price_only_pct": "Price-Only%",
        "single_bid_pct": "Single-Bidder%",
        "risk_pct": "Risk Score",
        "vako_disputes": "VAKO Disputes",
        "flags_str": "Flags",
    }
    st.dataframe(
        bp_top[list(buyer_cols.keys())].rename(columns=buyer_cols),
        width="stretch",
        hide_index=True,
    )

    # ---- Model performance expander ----
    with st.expander("Model Performance", expanded=False):
        st.markdown("**Score Distribution: Disputed vs Non-Disputed**")
        plot_df = deduped[["stage1_probability", "disputed"]].copy()
        plot_df["group"] = plot_df["disputed"].map({True: "Disputed", False: "Not Disputed"})

        fig = px.histogram(
            plot_df, x="stage1_probability", color="group",
            color_discrete_map={"Disputed": "#d32f2f", "Not Disputed": "#1976d2"},
            barmode="overlay", nbins=50, opacity=0.6,
            labels={"stage1_probability": "Risk Score", "group": ""},
        )
        fig.update_layout(height=300, margin=dict(t=10, b=30))
        st.plotly_chart(fig, width="stretch")

        n_total = len(deduped)
        n_disp = deduped["disputed"].sum()
        avg_disp = deduped.loc[deduped["disputed"], "stage1_probability"].mean()
        avg_non = deduped.loc[~deduped["disputed"], "stage1_probability"].mean()
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Procurements", f"{n_total:,}")
        mc2.metric("Disputed", f"{n_disp:,}")
        mc3.metric("Avg Score (Disputed)", f"{avg_disp:.4f}")
        mc4.metric("Avg Score (Non-Disputed)", f"{avg_non:.4f}")


# =========================================================================
# PAGE 3: Procurement Deep Dive
# =========================================================================
elif page == t("page_deep_dive"):

    st.title(t("dd_title"))

    # Back navigation
    if st.button(t("dd_back"), type="secondary"):
        st.session_state["navigate_to"] = "page_risk_monitor"
        st.rerun()

    # Load data
    months = available_months()
    latest_month = months[-1] if months else None
    latest_raw = load_monthly_results(latest_month) if latest_month else None
    features_dict = load_features()
    profiles = load_buyer_profiles()
    disputes_data = load_disputes()
    dispute_ids = set(disputes_data.get("disputes", {}).keys())
    v3 = load_v3_results()
    v3_lookup = {}
    if v3:
        for r in v3.get("results", []):
            v3_lookup[str(r["rhr_id"])] = r

    combined_lookup = {}
    for m in months:
        c = load_combined_results(m)
        if c:
            for r in c.get("results", []):
                combined_lookup[str(r["rhr_id"])] = r

    titles_data = load_procurement_titles()

    # Build selection options: top flagged from latest month (with titles)
    options = []
    if latest_raw:
        top_flagged = sorted(latest_raw["results"],
                             key=lambda x: x["stage1_probability"], reverse=True)[:30]
        for r in top_flagged:
            rid = str(r["rhr_id"])
            t_info = titles_data.get(rid, {})
            t_title = _clean_title(t_info.get("title", ""), 55)
            buyer_display = r["buyer_name"][:30] or t_info.get("buyer", "")[:30] or "Unknown"
            score_pct = f"{r['stage1_probability'] * 100:.1f}%"
            if t_title:
                lbl = f"{score_pct} \u2014 {t_title} ({buyer_display})"
            else:
                lbl = f"{score_pct} \u2014 {buyer_display} ({rid})"
            options.append((lbl, rid))

    # Check for navigation from Live Risk Monitor
    prefill_rhr = st.session_state.pop("deep_dive_rhr", "")

    # Selection UI
    sel_tab_top, sel_tab_search = st.tabs([t("dd_top_flagged"), t("dd_search_all")])

    with sel_tab_top:
        if options:
            # If navigated from another page, find the matching option index
            default_idx = 0
            if prefill_rhr:
                for i, (_, rid) in enumerate(options):
                    if rid == prefill_rhr:
                        default_idx = i
                        break
            chosen = st.selectbox(
                "Select from this month's highest-risk procurements",
                options,
                index=default_idx,
                format_func=lambda x: x[0],
            )
            selected_rhr = chosen[1]
        else:
            selected_rhr = ""

    with sel_tab_search:
        search_col1, search_col2 = st.columns([3, 1])
        with search_col1:
            search_query = st.text_input(
                t("dd_search_placeholder"),
                value=prefill_rhr if prefill_rhr and not options else "",
                placeholder="nt. 'Transpordiamet' / '285432'",
            )
        with search_col2:
            st.markdown("")  # spacing
            search_clicked = st.button("Search", type="primary")

        if search_query.strip():
            query = search_query.strip().lower()
            # Check if it's a direct RHR ID
            if query.isdigit() and query in features_dict:
                selected_rhr = query
            else:
                # Search through titles and buyers
                matches = []
                for rid, t_info in titles_data.items():
                    title_text = (t_info.get("title", "") or "").lower()
                    buyer_text = (t_info.get("buyer", "") or "").lower()
                    if query in title_text or query in buyer_text or query in rid:
                        feat_rec_search = features_dict.get(rid)
                        search_score = feat_rec_search.get("stage1_probability", 0) if feat_rec_search else 0
                        matches.append({
                            "rhr_id": rid,
                            "title": _clean_title(t_info.get("title", ""), 55),
                            "buyer": t_info.get("buyer", ""),
                            "score": search_score,
                        })
                if matches:
                    matches.sort(key=lambda x: -x["score"])
                    search_df = pd.DataFrame(matches[:20])
                    search_df["risk_pct"] = (search_df["score"] * 100).round(2).astype(str) + "%"
                    st.markdown(f"**{len(matches)} results** (showing top 20):")
                    search_event = st.dataframe(
                        search_df[["rhr_id", "risk_pct", "title", "buyer"]].rename(columns={
                            "rhr_id": "RHR ID", "risk_pct": "Risk", "title": "Title", "buyer": "Buyer",
                        }),
                        width="stretch", hide_index=True,
                        on_select="rerun", selection_mode="single-row",
                    )
                    if search_event and search_event.selection and search_event.selection.rows:
                        sel_idx = search_event.selection.rows[0]
                        if sel_idx < len(search_df):
                            selected_rhr = str(search_df.iloc[sel_idx]["rhr_id"])
                else:
                    st.caption("No matching procurements found.")

    if not selected_rhr:
        st.info("Select a procurement above or enter an RHR ID.")
        st.stop()

    # Find data for this procurement
    feat_rec = features_dict.get(selected_rhr)
    monthly_rec = None
    for m in months[::-1]:
        mr = load_monthly_results(m)
        if mr:
            for r in mr["results"]:
                if str(r["rhr_id"]) == selected_rhr:
                    monthly_rec = r
                    break
        if monthly_rec:
            break

    if not feat_rec and not monthly_rec:
        st.warning(f"No data found for RHR ID: {selected_rhr}")
        st.stop()

    is_disputed = selected_rhr in dispute_ids
    llm_result = combined_lookup.get(selected_rhr) or v3_lookup.get(selected_rhr)

    # ---- Procurement title & summary ----
    short_title, full_title = _get_procurement_title(selected_rhr, v3_lookup, disputes_data, titles_data)

    # Resolve buyer from all available sources
    buyer = (monthly_rec or {}).get("buyer_name") or (feat_rec or {}).get("buyer_name", "")
    if not buyer:
        buyer = titles_data.get(selected_rhr, {}).get("buyer", "")
    procedure = (monthly_rec or {}).get("procedure_type", "")
    contract = (monthly_rec or {}).get("contract_type", "")
    sector = (monthly_rec or {}).get("sector", "")
    value = (monthly_rec or {}).get("estimated_value")

    # Display title with styled header card
    _display_title = short_title if short_title else (
        f"{'Hange' if st.session_state.get('lang') == 'et' else 'Procurement'} {selected_rhr}"
    )
    _vako_label = "VAKO VAIDLUSTUS" if st.session_state.get("lang") == "et" else "VAKO DISPUTE"
    _disputed_badge = (
        '<span style="background: #fee2e2; color: #991b1b; padding: 2px 10px; '
        f'border-radius: 12px; font-size: 0.8rem; font-weight: 600; margin-left: 8px;">'
        f'{_vako_label}</span>'
    ) if is_disputed else ""

    st.markdown(
        f'<div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; '
        f'padding: 20px 24px; margin-bottom: 16px;">'
        f'<h3 style="margin: 0 0 8px 0; color: #0f172a !important;">{_display_title}{_disputed_badge}</h3>'
        f'<div style="display: flex; flex-wrap: wrap; gap: 12px; align-items: center; '
        f'color: #475569; font-size: 0.9rem;">'
        + (f'<span style="font-weight: 600; color: #1e3a5f;">{buyer}</span>' if buyer else "")
        + (f'<span>\u00b7</span><span>{_procedure_labels().get(procedure, procedure)}</span>' if procedure else "")
        + (f'<span>\u00b7</span><span>{contract.title()}</span>' if contract else "")
        + (f'<span>\u00b7</span><span>{_sector_labels().get(sector, sector)}</span>' if sector else "")
        + (f'<span>\u00b7</span><span style="font-weight: 600;">{fmt_eur(value)}</span>' if value else "")
        + f'<span>\u00b7</span><span style="font-family: monospace; font-size: 0.8rem;">{selected_rhr}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Show full title in expander if it was truncated
    if full_title and full_title != short_title and len(full_title) > 130:
        with st.expander("Full procurement description"):
            st.markdown(full_title)

    # Track feature contributions for summary
    _contrib_for_summary = None

    # ---- Two-column layout ----
    col_facts, col_contrib = st.columns([1, 1])

    with col_facts:
        st.subheader(t("dd_risk_score"))

        score = None
        if monthly_rec:
            score = monthly_rec["stage1_probability"]
        elif feat_rec and "stage1_probability" in feat_rec:
            score = feat_rec["stage1_probability"]

        # Evaluation criteria details
        if feat_rec:
            feat = feat_rec.get("features", {})
            pw = feat.get("price_weight", 0)
            qw = feat.get("quality_weight", 0)
            if pw or qw:
                total_w = pw + qw
                if total_w > 0:
                    pw_pct = pw / total_w * 100
                    qw_pct = qw / total_w * 100
                    st.markdown(f"**Evaluation:** Price {pw_pct:.0f}% · Quality {qw_pct:.0f}%")
                else:
                    st.markdown(f"**Price Weight:** {pw:.0%} · **Quality:** {qw:.0%}")

        # Additional details from features
        if feat_rec:
            flags = []
            if feat.get("is_eu_funded"):
                flags.append("EU funded")
            if feat.get("is_framework"):
                flags.append("Framework agreement")
            if feat.get("has_green"):
                flags.append("Green criteria")
            if feat.get("has_innovation"):
                flags.append("Innovation criteria")
            if feat.get("has_social"):
                flags.append("Social criteria")
            if flags:
                st.markdown("**Flags:** " + " · ".join(flags))

        if is_disputed:
            st.error("VAKO dispute(s) on record")

        # Risk gauge — capped at 30% to make the visual meaningful
        # (base rate is ~2%, so even 5% is elevated)
        if score is not None:
            color = risk_color(score)
            label = risk_label(score)
            display_pct = score * 100
            gauge_max = max(30, display_pct * 1.3)  # scale to show the score prominently
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=display_pct,
                number={"suffix": "%", "font": {"size": 40, "color": color}},
                title={"text": label.upper(), "font": {"size": 18, "color": color}},
                gauge={
                    "axis": {"range": [0, gauge_max], "ticksuffix": "%",
                             "tickvals": [0, 2, 4, 8, 15, gauge_max]},
                    "bar": {"color": color, "thickness": 0.75},
                    "steps": [
                        {"range": [0, 2], "color": "#e8f5e9"},
                        {"range": [2, 4], "color": "#f1f8e9"},
                        {"range": [4, 8], "color": "#fff9c4"},
                        {"range": [8, 15], "color": "#ffe0b2"},
                        {"range": [15, gauge_max], "color": "#ffcdd2"},
                    ],
                    "threshold": {
                        "line": {"color": "#666", "width": 2},
                        "value": 2,  # baseline dispute rate marker
                        "thickness": 0.8,
                    },
                },
            ))
            fig.update_layout(height=260, margin=dict(t=50, b=10, l=30, r=30))
            st.plotly_chart(fig, width="stretch")
            # Action-oriented guidance based on risk level
            _lang = st.session_state.get("lang", "en")
            if display_pct >= 15:
                _rec_label = "Soovitus:" if _lang == "et" else "Recommended:"
                _rec_text = ("Avaldamiseelne õiguslik ülevaade enne jätkamist. "
                             "Vaadake allpool olevat kontrollnimekirja konkreetsete tegevuste jaoks."
                             if _lang == "et" else
                             "Pre-publication legal review before proceeding. "
                             "Review the checklist below for specific actions.")
                st.markdown(
                    '<div style="background: #fef2f2; border-left: 4px solid #dc2626; '
                    'border-radius: 0 8px 8px 0; padding: 10px 14px; margin-top: 4px;">'
                    f'<strong style="color: #991b1b;">{_rec_label}</strong> '
                    f'<span style="color: #7f1d1d;">{_rec_text}</span></div>',
                    unsafe_allow_html=True,
                )
            elif display_pct >= 8:
                _rec_label = "Soovitus:" if _lang == "et" else "Recommended:"
                _rec_text = ("Vaadake üle riskitegurid ja kvaliteedihinnang allpool. "
                             "Käsitlege kontrollnimekirja kõrge prioriteediga punkte."
                             if _lang == "et" else
                             "Review the risk factors and quality assessment below. "
                             "Address high-priority items in the checklist.")
                st.markdown(
                    '<div style="background: #fff7ed; border-left: 4px solid #f97316; '
                    'border-radius: 0 8px 8px 0; padding: 10px 14px; margin-top: 4px;">'
                    f'<strong style="color: #9a3412;">{_rec_label}</strong> '
                    f'<span style="color: #7c2d12;">{_rec_text}</span></div>',
                    unsafe_allow_html=True,
                )
            elif display_pct >= 4:
                _note_label = "Märkus:" if _lang == "et" else "Note:"
                _note_text = ("Üle baastaseme riski. "
                              "Vaadake kvaliteedihinnangut parendusvõimaluste leidmiseks."
                              if _lang == "et" else
                              "Above baseline risk. "
                              "Check the quality assessment for improvement opportunities.")
                st.markdown(
                    '<div style="background: #fffbeb; border-left: 4px solid #f59e0b; '
                    'border-radius: 0 8px 8px 0; padding: 10px 14px; margin-top: 4px;">'
                    f'<strong style="color: #92400e;">{_note_label}</strong> '
                    f'<span style="color: #78350f;">{_note_text}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                _baseline_text = ("Baasline vaidlustuste määr on ~2%. Selle hanke skoor on alla tüüpilise riskitaseme."
                                  if _lang == "et" else
                                  "Baseline dispute rate is ~2%. This procurement scores below typical risk levels.")
                st.caption(_baseline_text)

    with col_contrib:
        st.subheader(t("dd_feature_contrib"))

        if feat_rec:
            model_data = load_model()
            model = model_data.get("model")
            scaler = model_data.get("scaler")
            feature_names = model_data.get("feature_names", [])

            if model and scaler and feature_names:
                feat = feat_rec.get("features", {})
                x_vals = [feat.get(fn, 0) for fn in feature_names]
                x_arr = np.array(x_vals).reshape(1, -1)
                x_scaled = scaler.transform(x_arr)
                coefs = model.coef_[0]
                contributions = coefs * x_scaled[0]

                contrib_df = pd.DataFrame({
                    "feature": feature_names,
                    "contribution": contributions,
                    "raw_value": x_vals,
                }).sort_values("contribution", key=abs, ascending=False).head(12)

                contrib_df["color"] = contrib_df["contribution"].apply(
                    lambda x: "#dc2626" if x > 0 else "#2563eb"
                )
                contrib_df = contrib_df.sort_values("contribution")

                contrib_df["label"] = contrib_df["feature"].apply(
                    lambda f: FEATURE_EXPLANATIONS.get(f, (f, ""))[0]
                )
                # Build hover text with explanation for each bar
                hover_texts = []
                for _, crow in contrib_df.iterrows():
                    fname = crow["feature"]
                    flabel, fexpl = FEATURE_EXPLANATIONS.get(fname, (fname, ""))
                    direction = "Increases" if crow["contribution"] > 0 else "Decreases"
                    # Wrap explanation to ~60 chars per line for tooltip
                    wrapped = "<br>".join(
                        fexpl[i:i + 60] for i in range(0, len(fexpl), 60)
                    ) if fexpl else ""
                    hover_texts.append(
                        f"<b>{flabel}</b> ({crow['contribution']:+.3f})<br>"
                        f"{direction} risk<br><br>{wrapped}"
                    )
                contrib_df["hover"] = hover_texts

                fig = go.Figure(go.Bar(
                    x=contrib_df["contribution"],
                    y=contrib_df["label"],
                    orientation="h",
                    marker_color=contrib_df["color"],
                    text=contrib_df["contribution"].apply(lambda x: f"{x:+.3f}"),
                    textposition="outside",
                    hovertext=contrib_df["hover"],
                    hoverinfo="text",
                ))
                fig.update_layout(
                    height=max(350, len(contrib_df) * 30 + 60),
                    margin=dict(t=10, b=30, l=10, r=10),
                    xaxis_title="Contribution to Risk Score",
                    hoverlabel=dict(bgcolor="white", font_size=12),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=True,
                               zerolinecolor="#94a3b8", zerolinewidth=1),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig, width="stretch")
                st.caption(
                    "Red bars increase risk, blue bars decrease it. "
                    "Hover over a bar for a detailed explanation."
                )

                # Store contrib_df for summary generation
                _contrib_for_summary = contrib_df
            else:
                _contrib_for_summary = None
                st.warning("Model data not available for feature contributions.")
        else:
            _contrib_for_summary = None
            st.info("No feature data available for this procurement.")

    # ---- Risk Summary ----
    summary_text = _generate_risk_summary(
        score, monthly_rec, feat_rec, _contrib_for_summary, llm_result, is_disputed
    )
    if summary_text:
        st.markdown("---")
        st.subheader(t("dd_risk_summary"))
        st.markdown(summary_text)

    # ---- Procurement Quality Assessment ----
    from pdf_report import compute_quality_assessment

    # Build integrity flags for quality assessment
    _integrity_data = load_integrity_lookups()
    _int_flags_for_qa = []
    if _integrity_data.get("donor_linked", {}).get(selected_rhr):
        _int_flags_for_qa.append(("Political Donor Link",
                                  "Board member donated \u22655K EUR to political party.", "high"))
    if _integrity_data.get("hidden_concentration", {}).get(selected_rhr):
        _int_flags_for_qa.append(("Hidden Ownership Concentration",
                                  "Shared beneficial owner with another winner at same buyer.", "high"))
    _zscore = _integrity_data.get("cpv_price_zscore", {}).get(selected_rhr)
    if _zscore is not None and _zscore > 2.0:
        _int_flags_for_qa.append(("CPV Price Anomaly",
                                  f"Value is {_zscore:.1f}\u03c3 above CPV-4 median.", "medium"))
    if _integrity_data.get("threshold_proximity", {}).get(selected_rhr):
        _int_flags_for_qa.append(("EU Threshold Proximity",
                                  "Value at 90-99% of EU threshold.", "medium"))
    _age = _integrity_data.get("winner_age_years", {}).get(selected_rhr)
    if _age is not None and _age < 2:
        _int_flags_for_qa.append(("Young Company",
                                  f"Winner was {_age:.1f} years old at award.", "medium"))

    _qa_features = (feat_rec or {}).get("features", {})
    quality_assessment = compute_quality_assessment(
        features=_qa_features,
        procedure=procedure,
        contract_type=contract,
        sector=sector,
        buyer_profile=profiles.get(buyer) if buyer else None,
        integrity_flags=_int_flags_for_qa,
    )

    st.markdown("---")
    st.subheader(t("dd_quality_score"))

    qa_score = quality_assessment["overall_score"]
    if qa_score >= 80:
        _qa_class = "excellent"
        _qa_label = t("quality_excellent")
        _qa_color = "#166534"
        _qa_bar_color = "#22c55e"
    elif qa_score >= 60:
        _qa_class = "good"
        _qa_label = t("quality_good")
        _qa_color = "#1e40af"
        _qa_bar_color = "#3b82f6"
    elif qa_score >= 40:
        _qa_class = "fair"
        _qa_label = t("quality_fair")
        _qa_color = "#92400e"
        _qa_bar_color = "#f59e0b"
    else:
        _qa_class = "poor"
        _qa_label = t("quality_needs_work")
        _qa_color = "#991b1b"
        _qa_bar_color = "#ef4444"

    # Overall score with visual bar
    st.markdown(
        f'<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 12px;">'
        f'<span class="quality-badge quality-{_qa_class}" style="font-size: 1.1rem;">'
        f'{qa_score}/100</span>'
        f'<span style="font-weight: 600; color: {_qa_color};">{_qa_label}</span>'
        f'<span style="color: #64748b; font-size: 0.9rem;">{quality_assessment["summary"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Visual progress bar for overall score
    st.markdown(
        f'<div style="background: #e2e8f0; border-radius: 8px; height: 12px; overflow: hidden; margin-bottom: 20px;">'
        f'<div style="background: {_qa_bar_color}; height: 100%; width: {qa_score}%; border-radius: 8px;'
        f' transition: width 0.5s;"></div></div>',
        unsafe_allow_html=True,
    )

    # Dimension scores with visual mini-bars
    dims = quality_assessment["dimensions"]
    dim_cols = st.columns(len(dims))
    _dim_colors = {
        "Competition & Access": "#3b82f6",
        "Criteria Quality": "#8b5cf6",
        "Strategic Value": "#06b6d4",
        "Transparency": "#22c55e",
        "Integrity & Governance": "#f59e0b",
    }
    _dim_translate = {
        "Competition & Access": t("qdim_competition"),
        "Criteria Quality": t("qdim_criteria"),
        "Strategic Value": t("qdim_strategy"),
        "Transparency": t("qdim_transparency"),
        "Integrity & Governance": t("qdim_integrity"),
    }
    for i, (dim_name, dim_data) in enumerate(dims.items()):
        s = dim_data["score"]
        m = dim_data["max"]
        pct = s / m * 100 if m > 0 else 0
        bar_col = _dim_colors.get(dim_name, "#64748b")
        _dim_display = _dim_translate.get(dim_name, dim_name)
        with dim_cols[i]:
            st.markdown(
                f'<div style="font-size: 0.78rem; color: #64748b; text-transform: uppercase; '
                f'letter-spacing: 0.3px; margin-bottom: 4px;">{_dim_display}</div>'
                f'<div style="font-size: 1.3rem; font-weight: 700; color: #0f172a;">{s}/{m}</div>'
                f'<div style="background: #e2e8f0; border-radius: 4px; height: 6px; '
                f'margin: 4px 0 4px 0; overflow: hidden;">'
                f'<div style="background: {bar_col}; height: 100%; width: {pct}%;"></div></div>'
                f'<div style="font-size: 0.75rem; color: #64748b;">'
                f'{dim_data["finding"][:50] if dim_data["finding"] else ""}</div>',
                unsafe_allow_html=True,
            )

    # Expandable dimension details
    with st.expander(t("dd_quality_breakdown"), expanded=False):
        for dim_name, dim_data in dims.items():
            s = dim_data["score"]
            m = dim_data["max"]
            pct = s / m * 100 if m > 0 else 0
            bar_col = _dim_colors.get(dim_name, "#64748b")
            st.markdown(
                f'<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">'
                f'<span style="font-weight: 600;">{_dim_translate.get(dim_name, dim_name)}</span>'
                f'<span style="color: #64748b;">({s}/{m})</span>'
                f'<div style="flex: 1; background: #e2e8f0; border-radius: 4px; height: 8px; '
                f'overflow: hidden;">'
                f'<div style="background: {bar_col}; height: 100%; width: {pct}%;"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(dim_data["explanation"])
            st.markdown("")

    # ---- VAKO Dispute Details (if disputed) ----
    if is_disputed:
        disp_details = _get_dispute_details(selected_rhr, disputes_data)
        if disp_details:
            st.markdown("---")
            st.subheader(t("dd_dispute_details"))
            for dd in disp_details:
                dc1, dc2, dc3 = st.columns(3)
                dc1.markdown(f"**Challenger:** {dd['challenger'] or 'Unknown'}")
                dc2.markdown(f"**Submitted:** {dd['submitted'] or 'Unknown'}")
                dc3.markdown(f"**Object:** {dd['object'] or 'Unknown'}")
                if dd.get("review_no"):
                    st.markdown(f"**Review No:** {dd['review_no']}")
                if dd.get("result"):
                    result_text = dd["result"]
                    if "rahuldamata" in result_text.lower():
                        st.success(f"**Result:** {result_text} (dispute rejected)")
                    elif "rahuldada" in result_text.lower():
                        st.error(f"**Result:** {result_text} (dispute sustained)")
                    else:
                        st.info(f"**Result:** {result_text}")
                elif dd.get("status"):
                    st.warning(f"**Status:** {dd['status']} (pending)")

    # ---- Integrity Flags ----
    integrity_data = load_integrity_lookups()
    int_flags = []
    if integrity_data.get("donor_linked", {}).get(selected_rhr):
        int_flags.append(("Political Donor Link",
                          "A board member of the winning company has donated \u22655,000 EUR "
                          "to a political party (ERJK records).", "high"))
    if integrity_data.get("hidden_concentration", {}).get(selected_rhr):
        int_flags.append(("Hidden Ownership Concentration",
                          "The winning company shares a beneficial owner with another company "
                          "that also won contracts from the same buyer.", "high"))
    zscore = integrity_data.get("cpv_price_zscore", {}).get(selected_rhr)
    if zscore is not None and zscore > 2.0:
        int_flags.append(("CPV Price Anomaly",
                          f"Contract value is {zscore:.1f}\u03c3 above the median for "
                          "its CPV-4 category.", "medium"))
    if integrity_data.get("threshold_proximity", {}).get(selected_rhr):
        pct = integrity_data["threshold_proximity"][selected_rhr]
        int_flags.append(("EU Threshold Proximity",
                          f"Contract value is at {pct * 100:.1f}% of an EU procurement "
                          "threshold (90-99% band).", "medium"))
    age = integrity_data.get("winner_age_years", {}).get(selected_rhr)
    if age is not None and age < 2:
        int_flags.append(("Young Company",
                          f"Winning company was only {age:.1f} years old at "
                          "time of contract award.", "medium"))

    if int_flags:
        st.markdown("---")
        st.subheader(t("dd_integrity_flags"))
        severity_icons = {"high": "\U0001f534", "medium": "\U0001f7e0", "low": "\U0001f7e2"}
        for flag_name, flag_desc, severity in int_flags:
            icon = severity_icons.get(severity, "\u26aa")
            st.markdown(f"{icon} **{flag_name}** \u2014 {flag_desc}")

    # ---- Actionable Checklist ----
    buyer_name = buyer or (monthly_rec or {}).get("buyer_name") or (feat_rec or {}).get("buyer_name", "")
    if not buyer_name:
        buyer_name = titles_data.get(selected_rhr, {}).get("buyer", "")
    profile = profiles.get(buyer_name)

    checklist = _generate_action_checklist(
        monthly_rec, feat_rec, llm_result, profile, _contrib_for_summary
    )
    if checklist:
        st.markdown("---")
        st.subheader(t("dd_checklist"))
        st.markdown(
            "Specific actions to reduce dispute risk for this procurement, "
            "based on its characteristics and historical patterns:"
        )
        priority_icons = {"HIGH": "\U0001f534", "MEDIUM": "\U0001f7e0", "LOW": "\U0001f7e2"}
        for priority, action, rationale in checklist:
            icon = priority_icons.get(priority, "\u26aa")
            with st.expander(f"{icon} **[{priority}]** {action}", expanded=(priority == "HIGH")):
                st.markdown(rationale)

    # ---- Buyer Profile with Sector Benchmarking ----
    if profile:
        st.markdown("---")
        st.subheader(f"{t('dd_buyer_profile')}: {buyer_name}")

        bc1, bc2, bc3, bc4, bc5 = st.columns(5)
        bc1.metric("Procurements", profile.get("procurement_count", 0))
        bc2.metric("Price-Only Rate", f"{profile.get('price_only_rate', 0):.0%}")
        bc3.metric("Single Bidder Rate", f"{profile.get('single_bidder_rate', 0):.0%}")
        bc4.metric("Avg Tenders", f"{profile.get('avg_tenders', 0):.1f}")
        bc5.metric("VAKO Disputes", profile.get("vako_disputes", 0))

        # Percentile ranking
        all_risk = sorted([p.get("risk_score", 0) for p in profiles.values()
                           if p.get("procurement_count", 0) >= 5])
        if all_risk:
            buyer_risk = profile.get("risk_score", 0)
            percentile = sum(1 for r in all_risk if r <= buyer_risk) / len(all_risk) * 100
            st.markdown(f"**Risk Percentile:** {percentile:.0f}th among {len(all_risk)} active buyers")

        flags = profile.get("risk_flags", [])
        if flags:
            st.markdown(f"**Risk Flags:** {', '.join(flags)}")

        # Sector benchmarking with radar chart
        if sector:
            benchmarks = _compute_sector_benchmarks(features_dict, disputes_data, sector)
            if benchmarks and benchmarks.get("total", 0) >= 10:
                st.markdown(f"**Buyer vs {_sector_labels().get(sector, sector)} Sector Average**")

                bm_cols = st.columns([1, 1])

                with bm_cols[0]:
                    # Radar chart comparing buyer to sector
                    categories = ["Competition", "Price-Only\nRate", "Dispute\nRate", "Experience"]
                    # Normalize metrics to 0-100 scale for radar
                    buyer_po = profile.get("price_only_rate", 0)
                    sector_po = benchmarks.get("price_only_rate", 0)
                    buyer_sb = profile.get("single_bidder_rate", 0)
                    buyer_disputes_pct = min(profile.get("vako_disputes", 0) / max(profile.get("procurement_count", 1), 1), 0.2)
                    sector_dr = benchmarks.get("dispute_rate", 0)
                    buyer_exp = min(profile.get("procurement_count", 0) / 100, 1.0)

                    # Higher is better for competition (inverse of single-bid), experience
                    # Lower is better for price-only, dispute rate
                    buyer_vals = [
                        (1 - buyer_sb) * 100,  # competition (inverse of single-bid)
                        (1 - buyer_po) * 100,   # quality criteria usage (inverse of price-only)
                        (1 - min(buyer_disputes_pct * 5, 1)) * 100,  # low disputes
                        buyer_exp * 100,  # experience
                    ]
                    sector_vals = [
                        70,  # average competition
                        (1 - sector_po) * 100,
                        (1 - min(sector_dr * 5, 1)) * 100,
                        50,  # average experience
                    ]

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=buyer_vals + [buyer_vals[0]],
                        theta=categories + [categories[0]],
                        fill="toself", fillcolor="rgba(37,99,235,0.15)",
                        line=dict(color="#2563eb", width=2),
                        name="This Buyer",
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=sector_vals + [sector_vals[0]],
                        theta=categories + [categories[0]],
                        fill="toself", fillcolor="rgba(148,163,184,0.1)",
                        line=dict(color="#94a3b8", width=1, dash="dot"),
                        name="Sector Avg",
                    ))
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
                        showlegend=True, height=280,
                        margin=dict(t=30, b=10, l=40, r=40),
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, x=0.3),
                    )
                    st.plotly_chart(fig_radar, width="stretch")

                with bm_cols[1]:
                    # Metric cards
                    _bm_data = [
                        ("Price-Only Rate", f"{buyer_po:.0%}", f"Sector: {sector_po:.0%}",
                         buyer_po > sector_po),
                        ("Single-Bidder Rate", f"{buyer_sb:.0%}", "EC red flag: >30%",
                         buyer_sb > 0.3),
                        ("Dispute Rate", f"{buyer_disputes_pct:.1%}", f"Sector: {sector_dr:.1%}",
                         buyer_disputes_pct > sector_dr),
                    ]
                    for bm_name, bm_val, bm_context, bm_warning in _bm_data:
                        _bm_border = "#f59e0b" if bm_warning else "#e2e8f0"
                        st.markdown(
                            f'<div style="background: #f8fafc; border: 1px solid {_bm_border}; '
                            f'border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;">'
                            f'<div style="font-size: 0.75rem; color: #64748b;">{bm_name}</div>'
                            f'<div style="font-size: 1.1rem; font-weight: 700;">{bm_val}</div>'
                            f'<div style="font-size: 0.7rem; color: #94a3b8;">{bm_context}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ---- LLM Analysis ----
    if llm_result:
        st.markdown("---")
        st.subheader(t("dd_ai_analysis"))

        # Combined results (stage 2)
        if "llm_score" in llm_result:
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("LLM Score", f"{llm_result['llm_score']}/10")
            conf = llm_result.get("llm_confidence", "\u2014")
            conf_colors = {"high": "red", "medium": "orange", "low": "green"}
            lc2.markdown(f"**Confidence:** :{conf_colors.get(conf, 'gray')}[{conf.upper()}]")
            lc3.metric("Issues Found", llm_result.get("llm_issues", "\u2014"))

            scenario = llm_result.get("llm_scenario", "")
            if scenario:
                st.info(f"**Most Likely Dispute Scenario:**\n\n{scenario}")

        # v3 deep analysis with document evidence
        if "pass2" in llm_result:
            p2 = llm_result["pass2"]

            issues = p2.get("specific_issues", [])
            if issues:
                st.markdown("#### Specific Issues Found")
                for i, issue in enumerate(issues, 1):
                    prob = issue.get("sustain_probability", "unknown")
                    prob_colors = {"high": "\U0001f534", "medium": "\U0001f7e0", "low": "\U0001f7e2"}
                    icon = prob_colors.get(prob, "\u26aa")

                    st.markdown(f"**{icon} Issue {i}: {issue.get('issue', 'Unknown issue')}**")

                    evidence = issue.get("evidence", "")
                    if evidence:
                        st.markdown("**From the procurement document:**")
                        st.markdown(f"> {evidence}")

                    rhs = issue.get("rhs_section", "")
                    if rhs:
                        st.caption(f"Legal basis: {rhs}")

                    precedent = issue.get("matching_precedent_pattern", "")
                    if precedent:
                        st.markdown(f"*Similar to VAKO precedent:* {precedent}")

                    st.markdown(f"*Probability of sustained challenge:* **{prob}**")

                    st.markdown(
                        "**Recommendation:** "
                        + _issue_recommendation(issue.get("issue", ""), evidence)
                    )
                    st.markdown("---")

            risk_factors = p2.get("risk_factors", [])
            mitigating = p2.get("mitigating_factors", [])

            if risk_factors or mitigating:
                rf_col, mf_col = st.columns(2)
                with rf_col:
                    st.markdown("#### Risk Factors")
                    for rf in risk_factors:
                        st.markdown(f"- \U0001f534 {rf}")
                with mf_col:
                    st.markdown("#### Mitigating Factors")
                    for mf in mitigating:
                        st.markdown(f"- \U0001f7e2 {mf}")

            justification = p2.get("score_justification", "")
            if justification:
                st.markdown(f"**Score Justification:** {justification}")

        # v3 pass1 document structure
        if "pass1" in llm_result:
            with st.expander("Document Structure Analysis", expanded=False):
                p1 = llm_result["pass1"]

                if p1.get("evaluation_criteria"):
                    st.markdown("**Evaluation Criteria:**")
                    for crit in p1["evaluation_criteria"]:
                        clarity = crit.get("clarity", "")
                        clarity_icon = {"clear": "\U0001f7e2", "unclear": "\U0001f534", "vague": "\U0001f7e0"}.get(clarity, "")
                        st.markdown(
                            f"- {crit.get('name', '?')}: **{crit.get('weight_percent', '?')}%** "
                            f"({crit.get('type', '?')}) {clarity_icon} {clarity}"
                        )

                if p1.get("qualification_requirements"):
                    st.markdown("**Qualification Requirements:**")
                    for qr in p1["qualification_requirements"]:
                        prop = qr.get("proportionality", "")
                        prop_icon = {
                            "proportionate": "\U0001f7e2",
                            "possibly_disproportionate": "\U0001f7e0",
                            "disproportionate": "\U0001f534",
                        }.get(prop, "")
                        st.markdown(f"- {prop_icon} {qr.get('requirement', '?')} \u2014 *{prop}*")

                if p1.get("restrictive_requirements"):
                    st.markdown("**Restrictive Requirements:**")
                    for rr in p1["restrictive_requirements"]:
                        sev = rr.get("severity", "")
                        sev_icon = {"high": "\U0001f534", "medium": "\U0001f7e0", "low": "\U0001f7e2"}.get(sev, "")
                        st.markdown(
                            f"- {sev_icon} **[{sev}]** {rr.get('requirement', '?')} "
                            f"({rr.get('type', '')})"
                        )

                docs = p1.get("key_documents_present", {})
                if docs:
                    st.markdown("**Key Documents:**")
                    for doc_name, present in docs.items():
                        icon = "\u2705" if present else "\u274c"
                        st.markdown(f"- {icon} {doc_name.replace('_', ' ').title()}")

    elif not llm_result:
        st.markdown("---")
        st.info(
            "No AI document analysis available for this procurement. "
            "Deep analysis is run on the highest-risk procurements each month."
        )

    # ---- Comparable Procurements ----
    if sector and procedure:
        st.markdown("---")
        st.subheader(t("dd_comparable"))
        st.markdown(
            f"Similar past procurements in **{_sector_labels().get(sector, sector)}** "
            f"using **{_procedure_labels().get(procedure, procedure).lower()}** procedure"
            + (f" in the **{fmt_eur(value)}** value range" if value else "")
            + ":"
        )

        comp_df = _find_comparable_procurements(
            features_dict, disputes_data, titles_data,
            sector, procedure, contract, value, selected_rhr,
        )
        if comp_df.empty:
            st.caption("No comparable procurements found with matching sector and procedure.")
        else:
            n_total = len(comp_df)
            n_disputed = comp_df["disputed"].sum()
            comp_rate = n_disputed / n_total if n_total else 0

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Comparable Found", f"{n_total}")
            cc2.metric("Of Which Disputed", f"{n_disputed}")
            cc3.metric("Dispute Rate", f"{comp_rate:.1%}")

            # Display table
            comp_display = comp_df.copy()
            comp_display["value_fmt"] = comp_display["value"].apply(fmt_eur)
            comp_display["dispute_flag"] = comp_display["disputed"].map({True: "DISPUTED", False: ""})
            comp_display["score_pct"] = (comp_display["score"] * 100).round(2).astype(str) + "%"

            show_cols = {
                "rhr_id": "RHR ID",
                "title": "Title",
                "buyer": "Buyer",
                "value_fmt": "Value",
                "score_pct": "Risk Score",
                "dispute_flag": "Status",
            }
            st.dataframe(
                comp_display[list(show_cols.keys())].rename(columns=show_cols),
                width="stretch",
                hide_index=True,
                height=min(len(comp_display) * 35 + 38, 400),
            )

            if n_disputed > 0:
                st.markdown(
                    f"**{comp_rate:.0%}** of comparable procurements were disputed. "
                    "Review the disputed cases for patterns that may apply to this procurement."
                )

    # ---- Extracted document text (for v3 procurements) ----
    doc_text = load_extracted_text(selected_rhr)
    if doc_text:
        st.markdown("---")
        st.subheader("Procurement Document Text")
        st.caption(f"{len(doc_text):,} characters extracted from procurement PDFs")

        if llm_result and "pass2" in llm_result:
            p2 = llm_result["pass2"]
            evidence_texts = []
            for issue in p2.get("specific_issues", []):
                ev = issue.get("evidence", "")
                if ev and len(ev) > 20:
                    evidence_texts.append((issue.get("issue", "Issue"), ev))

            if evidence_texts:
                st.markdown("#### Problematic Sections")
                for issue_name, evidence in evidence_texts:
                    st.markdown(f"**{issue_name}:**")
                    st.markdown(f"> {evidence}")
                    search_key = evidence[:50]
                    pos = doc_text.find(search_key)
                    if pos >= 0:
                        start = max(0, pos - 200)
                        end = min(len(doc_text), pos + len(evidence) + 200)
                        context = doc_text[start:end]
                        with st.expander("Show surrounding context from document"):
                            st.text(context)
                    st.markdown("")

        with st.expander("Full document text", expanded=False):
            st.text(doc_text[:5000])
            if len(doc_text) > 5000:
                st.caption(f"Showing first 5,000 of {len(doc_text):,} characters")

    # ---- PDF Report Download & External Links ----
    st.markdown("---")
    st.markdown(
        '<div style="background: linear-gradient(135deg, #0f1e3d, #1e3a5f); '
        'border-radius: 12px; padding: 20px 24px; margin-bottom: 16px;">'
        f'<h4 style="color: #fff !important; margin: 0 0 4px 0;">{t("dd_download_title")}</h4>'
        f'<p style="color: #94a3b8; font-size: 0.85rem; margin: 0;">'
        f'{t("dd_download_desc")}</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_pdf, col_link1, col_link2 = st.columns([2, 1, 1])

    with col_pdf:
        from pdf_report import generate_report as _gen_pdf

        # Build contributions list for PDF
        _pdf_contribs = []
        if _contrib_for_summary is not None:
            for _, crow in _contrib_for_summary.iterrows():
                fname = crow["feature"]
                flabel = FEATURE_EXPLANATIONS.get(fname, (fname, ""))[0]
                _pdf_contribs.append((fname, flabel, float(crow["contribution"])))

        # Build comparable procs for PDF
        _pdf_comparables = []
        if sector and procedure:
            _comp_df = _find_comparable_procurements(
                features_dict, disputes_data, titles_data,
                sector, procedure, contract, value, selected_rhr,
            )
            if not _comp_df.empty:
                for _, cr in _comp_df.iterrows():
                    _pdf_comparables.append({
                        "buyer": cr.get("buyer", ""),
                        "value_fmt": fmt_eur(cr.get("value")),
                        "score": cr.get("score", 0),
                        "disputed": cr.get("disputed", False),
                    })

        # Build dispute details for PDF
        _pdf_disputes = _get_dispute_details(selected_rhr, disputes_data) if is_disputed else []

        # Buyer profile for PDF
        _pdf_buyer_profile = profiles.get(buyer) if buyer else None
        _pdf_sector_bench = None
        if sector:
            _pdf_sector_bench = _compute_sector_benchmarks(features_dict, disputes_data, sector)

        # Build checklist for PDF
        _pdf_checklist = _generate_action_checklist(
            monthly_rec, feat_rec, llm_result, _pdf_buyer_profile, _contrib_for_summary
        )

        try:
            pdf_bytes = _gen_pdf(
                rhr_id=selected_rhr,
                title=short_title or f"Procurement {selected_rhr}",
                buyer=buyer or "Unknown",
                risk_score=score,
                risk_label_text=risk_label(score) if score else "Unknown",
                value_str=fmt_eur(value),
                procedure=procedure,
                procedure_label=_procedure_labels().get(procedure, procedure),
                contract_type=contract,
                sector=sector,
                sector_label=_sector_labels().get(sector, sector),
                features=_qa_features,
                feature_names=[],
                contributions=_pdf_contribs,
                integrity_flags=_int_flags_for_qa,
                checklist=_pdf_checklist,
                buyer_profile=_pdf_buyer_profile,
                sector_benchmarks=_pdf_sector_bench,
                comparable_procs=_pdf_comparables,
                dispute_details=_pdf_disputes,
                quality_assessment=quality_assessment,
            )
            st.download_button(
                label=f"\U0001f4c4 {t('dd_download_btn')}",
                data=pdf_bytes,
                file_name=f"procuresight_{selected_rhr}.pdf",
                mime="application/pdf",
                type="primary",
            )
        except Exception as e:
            st.caption(f"PDF generation unavailable: {e}")

    with col_link1:
        st.link_button(
            "View on riigihanked.riik.ee",
            f"https://riigihanked.riik.ee/rhr-web/#/procurement/{selected_rhr}/procurement-passport",
        )
    with col_link2:
        st.link_button(
            "View procurement documents",
            f"https://riigihanked.riik.ee/rhr-web/#/procurement/{selected_rhr}/documents",
        )


