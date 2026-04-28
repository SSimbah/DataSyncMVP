import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
# import anthropic
from openai import OpenAI
# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DataSync · Reconciliation Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Root theme */
:root {
    --bg: #0d0d0f;
    --surface: #16161a;
    --border: #2a2a35;
    --accent: #00e5a0;
    --accent2: #ff4d6d;
    --accent3: #ffd166;
    --text: #e8e8f0;
    --muted: #7a7a9a;
    --radius: 10px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}

/* App container */
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* ── Hero header ── */
.hero {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -1px;
    color: var(--text);
    margin: 0;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-size: 0.85rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0;
}

/* ── Section labels ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
}

/* ── Metric cards ── */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: var(--accent); }
.metric-card.red { border-left: 3px solid var(--accent2); }
.metric-card.green { border-left: 3px solid var(--accent); }
.metric-card.yellow { border-left: 3px solid var(--accent3); }
.metric-card.blue { border-left: 3px solid #4dabf7; }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.metric-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Status badge ── */
.badge {
    display: inline-block;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    padding: 0.2em 0.7em;
    border-radius: 20px;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-green  { background: rgba(0,229,160,0.15); color: var(--accent); }
.badge-red    { background: rgba(255,77,109,0.15); color: var(--accent2); }
.badge-yellow { background: rgba(255,209,102,0.15); color: var(--accent3); }
.badge-blue   { background: rgba(77,171,247,0.15); color: #4dabf7; }

/* ── AI summary box ── */
.ai-box {
    background: linear-gradient(135deg, #16161a 0%, #1a1a2e 100%);
    border: 1px solid #2a2a55;
    border-left: 3px solid var(--accent);
    border-radius: var(--radius);
    padding: 1.5rem;
    margin: 1.5rem 0;
    font-size: 0.92rem;
    line-height: 1.7;
    color: var(--text);
}
.ai-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

/* ── Upload zones ── */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1.5px dashed var(--border);
    border-radius: var(--radius);
    padding: 0.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent); }

/* ── Dataframes ── */
[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; }
.dataframe thead th {
    background: var(--surface) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    color: var(--accent) !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: 6px;
    padding: 0.55rem 1.5rem;
    font-weight: 700;
    transition: all 0.15s;
    cursor: pointer;
}
.stButton > button:hover {
    background: #00ffb3;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(0,229,160,0.3);
}
.stButton > button:active { transform: translateY(0); }

/* Selectbox, slider */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 6px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* Tabs */
[data-baseweb="tab-list"] { gap: 0.5rem; background: transparent !important; }
[data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--muted) !important;
    padding: 0.4rem 1rem !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: rgba(0,229,160,0.12) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}
[data-baseweb="tab-panel"] { padding-top: 1rem !important; }

/* Expander */
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* Dividers */
hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* Text input */
.stTextInput > div > div > input {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
}

/* Alerts */
.stSuccess { background: rgba(0,229,160,0.08) !important; border-left-color: var(--accent) !important; }
.stWarning { background: rgba(255,209,102,0.08) !important; }
.stError   { background: rgba(255,77,109,0.08)  !important; }
</style>
""", unsafe_allow_html=True)


# ── Helper: load file ─────────────────────────────────────────────────────────
@st.cache_data
def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported format. Please upload CSV or Excel.")
        return None


# ── Core comparison engine ────────────────────────────────────────────────────
def compare_datasets(src: pd.DataFrame, tgt: pd.DataFrame,
                     src_key: str, tgt_key: str,
                     field_map: list[tuple[str, str]]) -> dict:
    """
    Returns a dict with all comparison results.
    field_map: list of (src_col, tgt_col) tuples to compare.
    """
    results = {}

    # Rename target key to match source key for merging
    tgt_renamed = tgt.rename(columns={tgt_key: src_key})

    # ── Duplicate detection ──────────────────────────────────────────────────
    src_dupes = src[src.duplicated(subset=[src_key], keep=False)].copy()
    tgt_dupes = tgt[tgt.duplicated(subset=[tgt_key], keep=False)].copy()
    src_dupes["_issue"] = "Duplicate in Source"
    tgt_dupes["_issue"] = "Duplicate in Target"
    results["src_duplicates"] = src_dupes
    results["tgt_duplicates"] = tgt_dupes

    # ── Null / blank values ──────────────────────────────────────────────────
    src_null_report = []
    for col in src.columns:
        mask = src[col].isnull() | (src[col].astype(str).str.strip() == "")
        if mask.any():
            for idx, row in src[mask].iterrows():
                src_null_report.append({
                    "key": row[src_key], "dataset": "Source",
                    "column": col, "issue": "Null/Blank"
                })
    tgt_null_report = []
    for col in tgt.columns:
        mask = tgt[col].isnull() | (tgt[col].astype(str).str.strip() == "")
        if mask.any():
            for idx, row in tgt[mask].iterrows():
                tgt_null_report.append({
                    "key": row[tgt_key], "dataset": "Target",
                    "column": col, "issue": "Null/Blank"
                })
    results["null_issues"] = pd.DataFrame(src_null_report + tgt_null_report)

    # ── Missing / extra records ──────────────────────────────────────────────
    src_keys = set(src[src_key].dropna().astype(str))
    tgt_keys = set(tgt[tgt_key].dropna().astype(str))

    missing_in_target = src_keys - tgt_keys   # in source, not in target
    extra_in_target   = tgt_keys - src_keys   # in target, not in source

    results["missing_in_target"] = src[
        src[src_key].astype(str).isin(missing_in_target)
    ].copy()
    results["extra_in_target"] = tgt[
        tgt[tgt_key].astype(str).isin(extra_in_target)
    ].copy()

    # ── Field-level mismatches ───────────────────────────────────────────────
    if field_map:
        merged = pd.merge(
            src[[src_key] + [s for s, _ in field_map]],
            tgt_renamed[[src_key] + [t for _, t in field_map]],
            on=src_key,
            how="inner",
            suffixes=("_src", "_tgt"),
        )

        mismatch_rows = []
        for src_col, tgt_col in field_map:
            col_src = f"{src_col}_src"
            col_tgt = f"{tgt_col}_tgt"
            if col_src not in merged.columns:
                col_src = src_col
            if col_tgt not in merged.columns:
                col_tgt = tgt_col

            # Coerce to string for comparison (handles mixed types gracefully)
            s = merged[col_src].fillna("").astype(str).str.strip()
            t = merged[col_tgt].fillna("").astype(str).str.strip()
            diff = s != t
            for _, row in merged[diff].iterrows():
                mismatch_rows.append({
                    "key": row[src_key],
                    "source_field": src_col,
                    "target_field": tgt_col,
                    "source_value": row[col_src],
                    "target_value": row[col_tgt],
                })
        results["field_mismatches"] = pd.DataFrame(mismatch_rows)
    else:
        results["field_mismatches"] = pd.DataFrame()

    # ── Summary metrics ──────────────────────────────────────────────────────
    total_src     = len(src)
    total_tgt     = len(tgt)
    matched       = len(src_keys & tgt_keys)
    total_errors  = (
        len(results["missing_in_target"])
        + len(results["extra_in_target"])
        + len(results["src_duplicates"])
        + len(results["tgt_duplicates"])
        + len(results["field_mismatches"])
        + len(results["null_issues"])
    )
    match_pct = round((matched / max(len(src_keys), 1)) * 100, 1)

    results["summary"] = {
        "total_source":       total_src,
        "total_target":       total_tgt,
        "matched_records":    matched,
        "missing_in_target":  len(results["missing_in_target"]),
        "extra_in_target":    len(results["extra_in_target"]),
        "src_duplicates":     len(results["src_duplicates"]),
        "tgt_duplicates":     len(results["tgt_duplicates"]),
        "field_mismatches":   len(results["field_mismatches"]),
        "null_issues":        len(results["null_issues"]),
        "total_errors":       total_errors,
        "match_pct":          match_pct,
    }
    return results


# ── AI Summary via Anthropic ──────────────────────────────────────────────────
# def generate_ai_summary(summary: dict, api_key: str) -> str:
#     prompt = f"""You are a data quality analyst. Summarize the following dataset reconciliation results concisely in 3-5 sentences. Focus on the most critical issues, their potential business impact, and one actionable recommendation.

# Results:
# - Source records: {summary['total_source']}
# - Target records: {summary['total_target']}
# - Matched records: {summary['matched_records']} ({summary['match_pct']}%)
# - Missing in target: {summary['missing_in_target']}
# - Extra in target: {summary['extra_in_target']}
# - Source duplicates: {summary['src_duplicates']}
# - Target duplicates: {summary['tgt_duplicates']}
# - Field mismatches: {summary['field_mismatches']}
# - Null/blank issues: {summary['null_issues']}
# - Total errors: {summary['total_errors']}

# Write in plain professional language. No bullet points—prose only."""

#     client = anthropic.Anthropic(api_key=api_key)
#     msg = client.messages.create(
#         model="claude-opus-4-5",
#         max_tokens=400,
#         messages=[{"role": "user", "content": prompt}],
#     )
#     return msg.content[0].text
def generate_ai_summary(summary: dict, api_key: str) -> str:
    prompt = f"""You are a data quality analyst. Summarize the following dataset reconciliation results concisely in 3-5 sentences. Focus on the most critical issues, their potential business impact, and one actionable recommendation.

Results:
- Source records: {summary['total_source']}
- Target records: {summary['total_target']}
- Matched records: {summary['matched_records']} ({summary['match_pct']}%)
- Missing in target: {summary['missing_in_target']}
- Extra in target: {summary['extra_in_target']}
- Source duplicates: {summary['src_duplicates']}
- Target duplicates: {summary['tgt_duplicates']}
- Field mismatches: {summary['field_mismatches']}
- Null/blank issues: {summary['null_issues']}
- Total errors: {summary['total_errors']}

Write in plain professional language. No bullet points—prose only."""

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # fast + cheap (good for MVP)
        messages=[
            {"role": "system", "content": "You are a precise data quality analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )

    return response.choices[0].message.content

# ── Export helpers ────────────────────────────────────────────────────────────
def build_excel_report(results: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        summary_df = pd.DataFrame([results["summary"]])
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        sheets = {
            "Missing in Target":  results["missing_in_target"],
            "Extra in Target":    results["extra_in_target"],
            "Src Duplicates":     results["src_duplicates"],
            "Tgt Duplicates":     results["tgt_duplicates"],
            "Field Mismatches":   results["field_mismatches"],
            "Null Issues":        results["null_issues"],
        }
        for sheet, df in sheets.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="section-label">⚙ Configuration</p>', unsafe_allow_html=True)

    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for AI-powered findings summary",
    )

    st.markdown("---")
    st.markdown('<p class="section-label">📁 Upload Datasets</p>', unsafe_allow_html=True)

    src_file = st.file_uploader("Source Dataset", type=["csv", "xlsx", "xls"],
                                 key="src", label_visibility="collapsed",
                                 help="CSV or Excel")
    st.caption("↑ Source (CSV / Excel)")

    tgt_file = st.file_uploader("Target Dataset", type=["csv", "xlsx", "xls"],
                                 key="tgt", label_visibility="collapsed",
                                 help="CSV or Excel")
    st.caption("↑ Target (CSV / Excel)")

    st.markdown("---")
    st.markdown('<p class="section-label">ℹ About</p>', unsafe_allow_html=True)
    st.caption("DataSync v1.0 · Built with Streamlit + Pandas")
    st.caption("Compares two datasets and surfaces mismatches, duplicates, nulls and field-level differences.")


# ═══════════════════════════════════════════════════════════════════════════════
# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div>
    <h1 class="hero-title">Data<span>Sync</span></h1>
    <p class="hero-sub">Dataset Reconciliation &amp; Mismatch Analyzer</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── State guard ───────────────────────────────────────────────────────────────
if src_file is None or tgt_file is None:
    st.info("👈  Upload both datasets in the sidebar to get started.")
    st.stop()

src_df = load_file(src_file)
tgt_df = load_file(tgt_file)

if src_df is None or tgt_df is None:
    st.stop()

# ── Preview ───────────────────────────────────────────────────────────────────
with st.expander("🔍 Dataset Preview", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="section-label">Source</p>', unsafe_allow_html=True)
        st.caption(f"{len(src_df):,} rows · {len(src_df.columns)} columns")
        st.dataframe(src_df.head(20), use_container_width=True, height=220)
    with c2:
        st.markdown('<p class="section-label">Target</p>', unsafe_allow_html=True)
        st.caption(f"{len(tgt_df):,} rows · {len(tgt_df.columns)} columns")
        st.dataframe(tgt_df.head(20), use_container_width=True, height=220)

st.markdown("---")

# ── Field mapping ─────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">🔗 Key & Field Mapping</p>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    src_key = st.selectbox("Source Key Column", src_df.columns.tolist(), key="src_key",
                            help="The primary key / identifier in the source dataset")
with col_b:
    tgt_key = st.selectbox("Target Key Column", tgt_df.columns.tolist(), key="tgt_key",
                            help="The matching identifier in the target dataset")

st.caption("Map source → target fields to compare for value-level mismatches (optional).")

# Dynamic field-mapping rows
if "field_mappings" not in st.session_state:
    st.session_state.field_mappings = [("", "")]

def add_mapping():
    st.session_state.field_mappings.append(("", ""))

def remove_mapping(i):
    st.session_state.field_mappings.pop(i)

field_map: list[tuple[str, str]] = []
src_cols = ["(none)"] + [c for c in src_df.columns if c != src_key]
tgt_cols = ["(none)"] + [c for c in tgt_df.columns if c != tgt_key]

for i, (s_default, t_default) in enumerate(st.session_state.field_mappings):
    fc1, fc2, fc3 = st.columns([5, 5, 1])
    with fc1:
        sc = st.selectbox(f"Source field #{i+1}", src_cols,
                          index=src_cols.index(s_default) if s_default in src_cols else 0,
                          key=f"sf_{i}", label_visibility="collapsed")
    with fc2:
        tc = st.selectbox(f"Target field #{i+1}", tgt_cols,
                          index=tgt_cols.index(t_default) if t_default in tgt_cols else 0,
                          key=f"tf_{i}", label_visibility="collapsed")
    with fc3:
        if st.button("✕", key=f"rm_{i}", help="Remove mapping"):
            remove_mapping(i)
            st.rerun()
    if sc != "(none)" and tc != "(none)":
        field_map.append((sc, tc))

st.button("＋ Add field mapping", on_click=add_mapping)

st.markdown("---")

# ── Run comparison ────────────────────────────────────────────────────────────
run_col, _ = st.columns([2, 8])
with run_col:
    run = st.button("⚡ Run Comparison", use_container_width=True)

if run:
    with st.spinner("Comparing datasets…"):
        results = compare_datasets(src_df, tgt_df, src_key, tgt_key, field_map)
    st.session_state["results"] = results
    st.session_state["ran"] = True

if st.session_state.get("ran"):
    results = st.session_state["results"]
    s = results["summary"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    match_color = "#00e5a0" if s["match_pct"] >= 90 else ("#ffd166" if s["match_pct"] >= 70 else "#ff4d6d")

    st.markdown(f"""
    <div class="metrics-grid">
      <div class="metric-card green">
        <div class="metric-value" style="color:{match_color}">{s['match_pct']}%</div>
        <div class="metric-label">Match Rate</div>
      </div>
      <div class="metric-card blue">
        <div class="metric-value" style="color:#4dabf7">{s['matched_records']:,}</div>
        <div class="metric-label">Matched Records</div>
      </div>
      <div class="metric-card red">
        <div class="metric-value" style="color:#ff4d6d">{s['total_errors']:,}</div>
        <div class="metric-label">Total Errors</div>
      </div>
      <div class="metric-card yellow">
        <div class="metric-value" style="color:#ffd166">{s['missing_in_target']:,}</div>
        <div class="metric-label">Missing in Target</div>
      </div>
      <div class="metric-card yellow">
        <div class="metric-value" style="color:#ffd166">{s['extra_in_target']:,}</div>
        <div class="metric-label">Extra in Target</div>
      </div>
      <div class="metric-card red">
        <div class="metric-value" style="color:#ff4d6d">{s['src_duplicates'] + s['tgt_duplicates']:,}</div>
        <div class="metric-label">Duplicates</div>
      </div>
      <div class="metric-card red">
        <div class="metric-value" style="color:#ff4d6d">{s['field_mismatches']:,}</div>
        <div class="metric-label">Field Mismatches</div>
      </div>
      <div class="metric-card red">
        <div class="metric-value" style="color:#ff4d6d">{s['null_issues']:,}</div>
        <div class="metric-label">Null / Blank</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── AI summary ────────────────────────────────────────────────────────────
    if api_key:
        ai_col, _ = st.columns([3, 7])
        with ai_col:
            gen_ai = st.button("✨ Generate AI Summary")
        if gen_ai:
            with st.spinner("Claude is analysing results…"):
                try:
                    ai_text = generate_ai_summary(s, api_key)
                    st.session_state["ai_summary"] = ai_text
                except Exception as e:
                    st.error(f"AI error: {e}")

        if "ai_summary" in st.session_state:
            st.markdown(f"""
            <div class="ai-box">
              <div class="ai-header">✦ AI Analysis</div>
              {st.session_state['ai_summary']}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("💡 Enter an Anthropic API key in the sidebar to enable AI-powered analysis.")

    # ── Detailed results tabs ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">📋 Detailed Results</p>', unsafe_allow_html=True)

    tab_labels = [
        f"Missing ({s['missing_in_target']})",
        f"Extra ({s['extra_in_target']})",
        f"Duplicates ({s['src_duplicates'] + s['tgt_duplicates']})",
        f"Field Mismatches ({s['field_mismatches']})",
        f"Null / Blank ({s['null_issues']})",
    ]
    tabs = st.tabs(tab_labels)

    def show_tab(df, empty_msg="✅ No issues found."):
        if df is None or df.empty:
            st.success(empty_msg)
        else:
            st.dataframe(df, use_container_width=True, height=min(400, 50 + 35 * len(df)))

    with tabs[0]:
        st.caption("Records present in Source but absent in Target.")
        show_tab(results["missing_in_target"])

    with tabs[1]:
        st.caption("Records present in Target but absent in Source.")
        show_tab(results["extra_in_target"])

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Source duplicates")
            show_tab(results["src_duplicates"])
        with c2:
            st.caption("Target duplicates")
            show_tab(results["tgt_duplicates"])

    with tabs[3]:
        st.caption("Records with matching keys but differing field values.")
        show_tab(results["field_mismatches"])

    with tabs[4]:
        st.caption("Null or blank values detected in either dataset.")
        show_tab(results["null_issues"])

    # ── Export ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-label">💾 Export Report</p>', unsafe_allow_html=True)

    exp1, exp2 = st.columns(2)
    with exp1:
        excel_bytes = build_excel_report(results)
        st.download_button(
            "⬇ Download Excel Report",
            data=excel_bytes,
            file_name=f"datasync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with exp2:
        summary_json = json.dumps(s, indent=2)
        st.download_button(
            "⬇ Download Summary JSON",
            data=summary_json,
            file_name=f"datasync_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True,
        )
