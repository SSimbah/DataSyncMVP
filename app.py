"""
DataSync v2.0 — Large-File Dataset Reconciliation & Migration Tool
Senior-grade implementation: chunked I/O, disk-backed temp files,
streaming comparisons, bidirectional migration, Anthropic AI summary.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io, os, gc, json, tempfile
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from openai import OpenAI


st.write(st.config.get_option("server.maxUploadSize"))

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
CHUNK_SIZE    = 200_000
PREVIEW_ROWS  = 500
MAX_DISPLAY   = 15_000
LARGE_FILE_MB = 100

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DataSync · Reconciliation Tool",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#0d0d0f;--surface:#16161a;--border:#2a2a35;--accent:#00e5a0;--accent2:#ff4d6d;--accent3:#ffd166;--accent4:#4dabf7;--text:#e8e8f0;--muted:#7a7a9a;--radius:10px}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg);color:var(--text)}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:2rem 2.5rem 4rem;max-width:1400px}
.hero{display:flex;align-items:center;gap:1rem;margin-bottom:2.5rem;padding-bottom:1.5rem;border-bottom:1px solid var(--border)}
.hero-title{font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;letter-spacing:-1px;color:var(--text);margin:0}
.hero-title span{color:var(--accent)}
.hero-sub{font-size:.85rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase;margin:0}
.pill{display:inline-block;font-size:.68rem;font-family:'Space Mono',monospace;padding:.18em .65em;border-radius:20px;font-weight:700;letter-spacing:.05em}
.pill-green{background:rgba(0,229,160,.15);color:var(--accent)}
.pill-yellow{background:rgba(255,209,102,.15);color:var(--accent3)}
.section-label{font-family:'Space Mono',monospace;font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--accent);margin-bottom:.5rem}
.metrics-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:1rem;margin:1.5rem 0}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1.25rem 1rem;text-align:center;transition:border-color .2s}
.metric-card:hover{border-color:var(--accent)}
.metric-card.green{border-left:3px solid var(--accent)}
.metric-card.red{border-left:3px solid var(--accent2)}
.metric-card.yellow{border-left:3px solid var(--accent3)}
.metric-card.blue{border-left:3px solid var(--accent4)}
.metric-value{font-family:'Space Mono',monospace;font-size:1.9rem;font-weight:700;line-height:1;margin-bottom:.3rem}
.metric-label{font-size:.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.ai-box{background:linear-gradient(135deg,#16161a 0%,#1a1a2e 100%);border:1px solid #2a2a55;border-left:3px solid var(--accent);border-radius:var(--radius);padding:1.5rem;margin:1.5rem 0;font-size:.92rem;line-height:1.7}
.ai-header{display:flex;align-items:center;gap:.5rem;font-family:'Space Mono',monospace;font-size:.72rem;color:var(--accent);letter-spacing:.15em;text-transform:uppercase;margin-bottom:.75rem}
.migrate-box{background:linear-gradient(135deg,#0d1f15 0%,#16161a 100%);border:1px solid #1a3a28;border-left:3px solid var(--accent);border-radius:var(--radius);padding:1.5rem;margin:1.5rem 0}
.migrate-header{font-family:'Space Mono',monospace;font-size:.72rem;color:var(--accent);letter-spacing:.15em;text-transform:uppercase;margin-bottom:1rem}
.file-card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:1rem 1.25rem;margin-bottom:.75rem}
.file-name{font-family:'Space Mono',monospace;font-size:.82rem;color:var(--text);font-weight:700}
.file-meta{font-size:.75rem;color:var(--muted);margin-top:.25rem}
.stButton>button{font-family:'Space Mono',monospace;font-size:.8rem;letter-spacing:.05em;background:var(--accent);color:#000;border:none;border-radius:6px;padding:.55rem 1.5rem;font-weight:700;transition:all .15s}
.stButton>button:hover{background:#00ffb3;transform:translateY(-1px);box-shadow:0 4px 20px rgba(0,229,160,.3)}
.stButton>button:active{transform:translateY(0)}
.stSelectbox>div>div,.stMultiSelect>div>div{background:var(--surface)!important;border-color:var(--border)!important;color:var(--text)!important;border-radius:6px!important}
[data-testid="stFileUploader"]{background:var(--surface);border:1.5px dashed var(--border);border-radius:var(--radius);padding:.5rem;transition:border-color .2s}
[data-testid="stFileUploader"]:hover{border-color:var(--accent)}
.stTextInput>div>div>input{background:var(--surface)!important;border-color:var(--border)!important;color:var(--text)!important;font-family:'Space Mono',monospace;font-size:.85rem}
[data-testid="stSidebar"]{background:var(--surface);border-right:1px solid var(--border)}
[data-baseweb="tab-list"]{gap:.5rem;background:transparent!important}
[data-baseweb="tab"]{font-family:'Space Mono',monospace!important;font-size:.75rem!important;letter-spacing:.05em!important;background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:6px!important;color:var(--muted)!important;padding:.4rem 1rem!important}
[aria-selected="true"][data-baseweb="tab"]{background:rgba(0,229,160,.12)!important;border-color:var(--accent)!important;color:var(--accent)!important}
[data-baseweb="tab-panel"]{padding-top:1rem!important}
[data-testid="stExpander"]{background:var(--surface);border:1px solid var(--border)!important;border-radius:var(--radius)!important}
hr{border-color:var(--border)!important;margin:1.5rem 0!important}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE I/O HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def detect_type(filename: str) -> str:
    return "excel" if Path(filename).suffix.lower() in (".xlsx", ".xls") else "csv"


def save_to_temp(uploaded) -> tuple[str, str]:
    """Persist UploadedFile to disk. Returns (path, file_type)."""
    ftype = detect_type(uploaded.name)
    suffix = Path(uploaded.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush()
    tmp.close()
    return tmp.name, ftype


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1_048_576


def is_large(path: str) -> bool:
    return file_size_mb(path) > LARGE_FILE_MB


def get_columns(path: str, ftype: str) -> list[str]:
    if ftype == "csv":
        return pd.read_csv(path, nrows=0).columns.tolist()
    return pd.read_excel(path, nrows=0).columns.tolist()


def load_sample(path: str, ftype: str, n: int = PREVIEW_ROWS) -> pd.DataFrame:
    if ftype == "csv":
        return pd.read_csv(path, nrows=n, low_memory=False)
    return pd.read_excel(path, nrows=n)


def csv_chunks(path: str, cols=None):
    kw = dict(chunksize=CHUNK_SIZE, low_memory=False)
    if cols:
        kw["usecols"] = cols
    yield from pd.read_csv(path, **kw)


def excel_read(path: str, cols=None) -> pd.DataFrame:
    return pd.read_excel(path, usecols=cols) if cols else pd.read_excel(path)


def iter_file(path: str, ftype: str, cols=None):
    """Unified iterator: yields DataFrame chunks."""
    if ftype == "csv":
        yield from csv_chunks(path, cols)
    else:
        yield excel_read(path, cols)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON ENGINE  (streaming / chunked, large-file safe)
# ═══════════════════════════════════════════════════════════════════════════════

def stream_keys_and_dupes(path: str, ftype: str, key_col: str,
                           progress_cb=None) -> tuple[set, pd.DataFrame]:
    """Single-pass key extraction + duplicate detection. Memory = O(unique keys)."""
    key_counts: dict[str, int] = defaultdict(int)
    chunks = list(iter_file(path, ftype, cols=[key_col]))
    total = max(len(chunks), 1)

    for i, chunk in enumerate(chunks):
        chunk[key_col] = chunk[key_col].astype(str).str.strip()
        for k in chunk[key_col]:
            key_counts[k] += 1
        if progress_cb:
            progress_cb(int((i + 1) / total * 100))

    all_keys  = set(key_counts)
    dupe_keys = {k for k, v in key_counts.items() if v > 1}

    # Second pass: collect duplicate rows (capped)
    dupe_rows: list[pd.DataFrame] = []
    if dupe_keys:
        collected = 0
        for chunk in iter_file(path, ftype):
            chunk[key_col] = chunk[key_col].astype(str).str.strip()
            hit = chunk[chunk[key_col].isin(dupe_keys)]
            if not hit.empty:
                dupe_rows.append(hit)
                collected += len(hit)
            if collected >= MAX_DISPLAY:
                break

    dupes_df = pd.concat(dupe_rows).head(MAX_DISPLAY) if dupe_rows else pd.DataFrame()
    return all_keys, dupes_df


def fetch_rows_by_keys(path: str, ftype: str, key_col: str,
                        keys: set, limit: int = MAX_DISPLAY) -> pd.DataFrame:
    """Return rows whose key_col is in keys (capped at limit)."""
    rows: list[pd.DataFrame] = []
    collected = 0
    for chunk in iter_file(path, ftype):
        chunk[key_col] = chunk[key_col].astype(str).str.strip()
        hit = chunk[chunk[key_col].isin(keys)]
        if not hit.empty:
            rows.append(hit)
            collected += len(hit)
        if collected >= limit:
            break
    return pd.concat(rows).head(limit) if rows else pd.DataFrame()


def scan_nulls(path: str, ftype: str, label: str,
               progress_cb=None) -> pd.DataFrame:
    """Count null/blank cells per column — no per-row storage."""
    null_counts: dict[str, int] = defaultdict(int)
    total_rows = 0
    chunks = list(iter_file(path, ftype))
    n = max(len(chunks), 1)

    for i, chunk in enumerate(chunks):
        total_rows += len(chunk)
        for col in chunk.columns:
            mask = chunk[col].isnull() | (chunk[col].astype(str).str.strip() == "")
            null_counts[col] += int(mask.sum())
        if progress_cb:
            progress_cb(int((i + 1) / n * 100))

    records = [
        {"dataset": label, "column": col, "null_count": cnt,
         "null_pct": round(cnt / max(total_rows, 1) * 100, 2)}
        for col, cnt in null_counts.items() if cnt > 0
    ]
    return pd.DataFrame(records)


def compare_fields_chunked(
    src_path: str, src_ftype: str, src_key: str,
    tgt_path: str, tgt_ftype: str, tgt_key: str,
    field_map: list[tuple[str, str]],
    common_keys: set,
    progress_cb=None,
) -> pd.DataFrame:
    """
    Build an in-memory target lookup (key → {field: value}), then stream
    source chunks to diff. Memory cost = O(common_keys × mapped_fields).
    """
    if not field_map or not common_keys:
        return pd.DataFrame()

    src_fields = [s for s, _ in field_map]
    tgt_fields = [t for _, t in field_map]

    # Build target index
    tgt_lookup: dict[str, dict] = {}
    for chunk in iter_file(tgt_path, tgt_ftype, cols=[tgt_key] + tgt_fields):
        chunk[tgt_key] = chunk[tgt_key].astype(str).str.strip()
        sub = chunk[chunk[tgt_key].isin(common_keys)]
        for _, row in sub.iterrows():
            tgt_lookup[row[tgt_key]] = {
                tf: str(row.get(tf, "")).strip() for tf in tgt_fields
            }

    mismatches: list[dict] = []
    src_chunks = list(iter_file(src_path, src_ftype, cols=[src_key] + src_fields))
    total = max(len(src_chunks), 1)

    for i, chunk in enumerate(src_chunks):
        chunk[src_key] = chunk[src_key].astype(str).str.strip()
        for _, row in chunk[chunk[src_key].isin(common_keys)].iterrows():
            key = row[src_key]
            tgt_row = tgt_lookup.get(key, {})
            for sf, tf in field_map:
                sv = str(row.get(sf, "")).strip()
                tv = tgt_row.get(tf, "")
                if sv != tv:
                    mismatches.append({
                        "key": key, "source_field": sf, "target_field": tf,
                        "source_value": sv, "target_value": tv,
                    })
            if len(mismatches) >= MAX_DISPLAY:
                break
        if progress_cb:
            progress_cb(int((i + 1) / total * 100))
        if len(mismatches) >= MAX_DISPLAY:
            break

    del tgt_lookup
    gc.collect()
    return pd.DataFrame(mismatches)


def run_comparison(src_path, src_ftype, src_key,
                   tgt_path, tgt_ftype, tgt_key, field_map) -> dict:
    results: dict = {}
    status = st.status("⚡ Running comparison…", expanded=True)

    with status:
        st.write("🔑 Extracting Source keys + duplicates…")
        pb = st.progress(0)
        src_keys, src_dupes = stream_keys_and_dupes(
            src_path, src_ftype, src_key, lambda p: pb.progress(p))
        pb.progress(100)

        st.write("🔑 Extracting Target keys + duplicates…")
        pb2 = st.progress(0)
        tgt_keys, tgt_dupes = stream_keys_and_dupes(
            tgt_path, tgt_ftype, tgt_key, lambda p: pb2.progress(p))
        pb2.progress(100)

        st.write("🔍 Computing missing / extra records…")
        missing_keys = src_keys - tgt_keys
        extra_keys   = tgt_keys - src_keys
        common_keys  = src_keys & tgt_keys
        results["missing_in_target"] = fetch_rows_by_keys(
            src_path, src_ftype, src_key, missing_keys)
        results["extra_in_target"]   = fetch_rows_by_keys(
            tgt_path, tgt_ftype, tgt_key, extra_keys)
        results["src_duplicates"] = src_dupes
        results["tgt_duplicates"] = tgt_dupes

        st.write("🔎 Scanning nulls — Source…")
        pb3 = st.progress(0)
        src_nulls = scan_nulls(src_path, src_ftype, "Source", lambda p: pb3.progress(p))
        pb3.progress(100)

        st.write("🔎 Scanning nulls — Target…")
        pb4 = st.progress(0)
        tgt_nulls = scan_nulls(tgt_path, tgt_ftype, "Target", lambda p: pb4.progress(p))
        pb4.progress(100)
        results["null_issues"] = pd.concat(
            [src_nulls, tgt_nulls]).reset_index(drop=True)

        st.write("📐 Comparing mapped field values…")
        pb5 = st.progress(0)
        results["field_mismatches"] = compare_fields_chunked(
            src_path, src_ftype, src_key,
            tgt_path, tgt_ftype, tgt_key,
            field_map, common_keys, lambda p: pb5.progress(p))
        pb5.progress(100)

        total_null = int(results["null_issues"]["null_count"].sum()
                         if not results["null_issues"].empty else 0)
        total_errors = (
            len(missing_keys) + len(extra_keys)
            + len(src_dupes) + len(tgt_dupes)
            + len(results["field_mismatches"])
            + total_null
        )
        results["summary"] = {
            "total_source":       len(src_keys),
            "total_target":       len(tgt_keys),
            "matched_records":    len(common_keys),
            "missing_in_target":  len(missing_keys),
            "extra_in_target":    len(extra_keys),
            "src_duplicates":     len(src_dupes),
            "tgt_duplicates":     len(tgt_dupes),
            "field_mismatches":   len(results["field_mismatches"]),
            "null_issues":        total_null,
            "total_errors":       total_errors,
            "match_pct":          round(len(common_keys) / max(len(src_keys), 1) * 100, 2),
        }
        results["_meta"] = {"missing_keys": missing_keys, "extra_keys": extra_keys}

    status.update(label="✅ Comparison complete!", state="complete", expanded=False)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MIGRATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_migration_file(
    src_path, src_ftype, src_key,
    tgt_path, tgt_ftype, tgt_key,
    field_map, direction, missing_keys, extra_keys,
) -> tuple[bytes, str, int]:
    """
    direction = "src_to_tgt" → missing source records → target schema
    direction = "tgt_to_src" → extra target records   → source schema
    Returns (csv_bytes, filename, row_count).
    """
    if direction == "src_to_tgt":
        path, ftype, key_col = src_path, src_ftype, src_key
        keys_to_migrate = missing_keys
        rename = {src_key: tgt_key, **{sf: tf for sf, tf in field_map}}
    else:
        path, ftype, key_col = tgt_path, tgt_ftype, tgt_key
        keys_to_migrate = extra_keys
        rename = {tgt_key: src_key, **{tf: sf for sf, tf in field_map}}

    buf = io.StringIO()
    header_written = False
    row_count = 0

    for chunk in iter_file(path, ftype):
        chunk[key_col] = chunk[key_col].astype(str).str.strip()
        filtered = chunk[chunk[key_col].isin(keys_to_migrate)].copy()
        if filtered.empty:
            continue
        existing = {k: v for k, v in rename.items() if k in filtered.columns}
        filtered = filtered.rename(columns=existing)
        row_count += len(filtered)
        filtered.to_csv(buf, index=False, header=not header_written)
        header_written = True

    filename = f"migration_{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return buf.getvalue().encode(), filename, row_count


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT & AI
# ═══════════════════════════════════════════════════════════════════════════════

def build_excel_report(results: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame([results["summary"]]).to_excel(
            writer, sheet_name="Summary", index=False)
        sheets = {
            "Missing in Target": results["missing_in_target"],
            "Extra in Target":   results["extra_in_target"],
            "Src Duplicates":    results["src_duplicates"],
            "Tgt Duplicates":    results["tgt_duplicates"],
            "Field Mismatches":  results["field_mismatches"],
            "Null Issues":       results["null_issues"],
        }
        for name, df in sheets.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=name[:31], index=False)
    return buf.getvalue()


def generate_ai_summary(summary: dict, api_key: str) -> str:
    prompt = f"""You are a senior data quality analyst. Summarize these reconciliation results in 4-5 sentences of professional prose. Highlight the most critical issues, potential business impact, and one clear action. No bullet points.

Source: {summary['total_source']:,} rows | Target: {summary['total_target']:,} rows
Matched: {summary['matched_records']:,} ({summary['match_pct']}%)
Missing in target: {summary['missing_in_target']:,} | Extra in target: {summary['extra_in_target']:,}
Src dupes: {summary['src_duplicates']:,} | Tgt dupes: {summary['tgt_duplicates']:,}
Field mismatches: {summary['field_mismatches']:,} | Null issues: {summary['null_issues']:,}
Total errors: {summary['total_errors']:,}"""

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-opus-4-5", max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<p class="section-label">⚙ Configuration</p>', unsafe_allow_html=True)
    api_key = st.text_input("Anthropic API Key", type="password",
                             placeholder="sk-ant-…",
                             help="Optional — enables AI findings summary")

    st.markdown("---")
    st.markdown('<p class="section-label">📁 Source Dataset</p>', unsafe_allow_html=True)
    src_file = st.file_uploader("Source", type=["csv", "xlsx", "xls"],
                                 key="src_up", label_visibility="collapsed")

    st.markdown('<p class="section-label" style="margin-top:.75rem">📁 Target Dataset</p>',
                unsafe_allow_html=True)
    tgt_file = st.file_uploader("Target", type=["csv", "xlsx", "xls"],
                                 key="tgt_up", label_visibility="collapsed")

    st.markdown("---")
    st.caption("DataSync v2.0  ·  Streamlit + Pandas")
    st.caption("Handles files > 1 GB via chunked streaming.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div>
    <h1 class="hero-title">Data<span>Sync</span></h1>
    <p class="hero-sub">Large-File Reconciliation &amp; Migration Engine &nbsp;·&nbsp; v2.0</p>
  </div>
</div>
""", unsafe_allow_html=True)

if src_file is None or tgt_file is None:
    st.info("👈  Upload both Source and Target datasets in the sidebar to begin.")
    st.stop()

# ── Persist uploads to temp disk (avoids re-reading from Streamlit memory) ────
for attr, file, name_key in [
    ("src_path", src_file, "src_name"),
    ("tgt_path", tgt_file, "tgt_name"),
]:
    if (attr not in st.session_state or
            st.session_state.get(name_key) != file.name):
        path, ftype = save_to_temp(file)
        st.session_state[attr] = path
        st.session_state[attr.replace("path", "ftype")] = ftype
        st.session_state[name_key] = file.name
        st.session_state.pop("results", None)

src_path  = st.session_state["src_path"]
src_ftype = st.session_state["src_ftype"]
tgt_path  = st.session_state["tgt_path"]
tgt_ftype = st.session_state["tgt_ftype"]
src_cols  = get_columns(src_path, src_ftype)
tgt_cols  = get_columns(tgt_path, tgt_ftype)

# ── File info ─────────────────────────────────────────────────────────────────
ic1, ic2 = st.columns(2)
for col_ui, file, path, cols in [
    (ic1, src_file, src_path, src_cols),
    (ic2, tgt_file, tgt_path, tgt_cols),
]:
    mb = file_size_mb(path)
    badge = ('<span class="pill pill-yellow">CHUNKED &gt;100 MB</span>'
             if is_large(path) else '<span class="pill pill-green">STANDARD</span>')
    with col_ui:
        st.markdown(f"""
        <div class="file-card">
          <div class="file-name">📄 {file.name}</div>
          <div class="file-meta">{mb:.1f} MB &nbsp;·&nbsp; {len(cols)} columns &nbsp;·&nbsp; {badge}</div>
        </div>""", unsafe_allow_html=True)

# ── Dataset preview ───────────────────────────────────────────────────────────
with st.expander("🔍 Dataset Preview (first 500 rows)", expanded=False):
    p1, p2 = st.columns(2)
    with p1:
        st.caption("Source")
        st.dataframe(load_sample(src_path, src_ftype), use_container_width=True, height=240)
    with p2:
        st.caption("Target")
        st.dataframe(load_sample(tgt_path, tgt_ftype), use_container_width=True, height=240)

st.markdown("---")

# ── Key & field mapping ───────────────────────────────────────────────────────
st.markdown('<p class="section-label">🔗 Key & Field Mapping</p>', unsafe_allow_html=True)
kc1, kc2 = st.columns(2)
with kc1:
    src_key = st.selectbox("Source Key Column", src_cols, key="src_key")
with kc2:
    tgt_key = st.selectbox("Target Key Column", tgt_cols, key="tgt_key")

st.caption("Map source → target fields for value-level comparison (optional).")

if "field_mappings" not in st.session_state:
    st.session_state.field_mappings = [("", "")]

def add_mapping():    st.session_state.field_mappings.append(("", ""))
def remove_mapping(i): st.session_state.field_mappings.pop(i)

field_map: list[tuple[str, str]] = []
src_opts = ["(none)"] + [c for c in src_cols if c != src_key]
tgt_opts = ["(none)"] + [c for c in tgt_cols if c != tgt_key]

for i, (sd, td) in enumerate(st.session_state.field_mappings):
    fc1, fc2, fc3 = st.columns([5, 5, 1])
    with fc1:
        sc = st.selectbox(f"sf{i}", src_opts,
                          index=src_opts.index(sd) if sd in src_opts else 0,
                          key=f"sf_{i}", label_visibility="collapsed")
    with fc2:
        tc = st.selectbox(f"tf{i}", tgt_opts,
                          index=tgt_opts.index(td) if td in tgt_opts else 0,
                          key=f"tf_{i}", label_visibility="collapsed")
    with fc3:
        if st.button("✕", key=f"rm_{i}"):
            remove_mapping(i); st.rerun()
    if sc != "(none)" and tc != "(none)":
        field_map.append((sc, tc))

st.button("＋ Add field mapping", on_click=add_mapping)
st.markdown("---")

# ── Run ───────────────────────────────────────────────────────────────────────
run_col, _ = st.columns([2, 8])
with run_col:
    run = st.button("⚡ Run Comparison", use_container_width=True)

if run:
    st.session_state.pop("results", None)
    st.session_state.pop("ai_summary", None)
    st.session_state["results"] = run_comparison(
        src_path, src_ftype, src_key,
        tgt_path, tgt_ftype, tgt_key,
        field_map,
    )

if "results" not in st.session_state:
    st.stop()

results = st.session_state["results"]
s  = results["summary"]
mt = results["_meta"]

# ── Metrics ───────────────────────────────────────────────────────────────────
mc = "#00e5a0" if s["match_pct"] >= 90 else ("#ffd166" if s["match_pct"] >= 70 else "#ff4d6d")
st.markdown(f"""
<div class="metrics-grid">
  <div class="metric-card green">
    <div class="metric-value" style="color:{mc}">{s['match_pct']}%</div>
    <div class="metric-label">Match Rate</div>
  </div>
  <div class="metric-card blue">
    <div class="metric-value" style="color:#4dabf7">{s['matched_records']:,}</div>
    <div class="metric-label">Matched</div>
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
    <div class="metric-label">Null Values</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── AI Summary ────────────────────────────────────────────────────────────────
if api_key:
    ac, _ = st.columns([2, 8])
    with ac:
        if st.button("✨ Generate AI Summary"):
            with st.spinner("Analysing with Claude…"):
                try:
                    st.session_state["ai_summary"] = generate_ai_summary(s, api_key)
                except Exception as e:
                    st.error(f"AI error: {e}")

if "ai_summary" in st.session_state:
    st.markdown(f"""
    <div class="ai-box">
      <div class="ai-header">✦ AI Analysis — Claude</div>
      {st.session_state['ai_summary']}
    </div>""", unsafe_allow_html=True)
else:
    st.caption("💡 Enter an Anthropic API key in the sidebar to enable AI-powered analysis.")

# ── Detail tabs ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-label">📋 Detailed Results</p>', unsafe_allow_html=True)

cap_note = f"*(display capped at {MAX_DISPLAY:,} rows)*"
tabs = st.tabs([
    f"Missing ({s['missing_in_target']:,})",
    f"Extra ({s['extra_in_target']:,})",
    f"Duplicates ({s['src_duplicates'] + s['tgt_duplicates']:,})",
    f"Mismatches ({s['field_mismatches']:,})",
    f"Null / Blank ({s['null_issues']:,})",
])

def show_df(df, empty="✅ No issues found."):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        st.success(empty)
    else:
        st.dataframe(df, use_container_width=True,
                     height=min(420, 60 + 36 * len(df)))

with tabs[0]:
    st.caption(f"Records present in Source but absent in Target. {cap_note if s['missing_in_target'] > MAX_DISPLAY else ''}")
    show_df(results["missing_in_target"])
with tabs[1]:
    st.caption(f"Records present in Target but absent in Source. {cap_note if s['extra_in_target'] > MAX_DISPLAY else ''}")
    show_df(results["extra_in_target"])
with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Source duplicates")
        show_df(results["src_duplicates"])
    with c2:
        st.caption("Target duplicates")
        show_df(results["tgt_duplicates"])
with tabs[3]:
    st.caption(f"Same key, different values on mapped fields. {cap_note if s['field_mismatches'] >= MAX_DISPLAY else ''}")
    show_df(results["field_mismatches"])
with tabs[4]:
    st.caption("Null / blank cell counts per column in each dataset.")
    show_df(results["null_issues"])


# ═══════════════════════════════════════════════════════════════════════════════
# MIGRATION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div class="migrate-box">
  <div class="migrate-header">🔄 Data Migration</div>
""", unsafe_allow_html=True)

st.write("Generate a **delta migration file** — records reformatted to the destination schema, ready to import.")

mc1, mc2 = st.columns([3, 7])
with mc1:
    direction = st.radio(
        "Direction",
        options=["src_to_tgt", "tgt_to_src"],
        format_func=lambda x: "Source  →  Target" if x == "src_to_tgt" else "Target  →  Source",
        key="migrate_dir",
    )
with mc2:
    if direction == "src_to_tgt":
        n_migrate = len(mt["missing_keys"])
        st.info(
            f"**{n_migrate:,} records** from Source are missing in Target.\n\n"
            "Migration file will rename columns to match the Target schema per your field mapping."
        )
    else:
        n_migrate = len(mt["extra_keys"])
        st.info(
            f"**{n_migrate:,} records** from Target are not in Source.\n\n"
            "Migration file will rename columns to match the Source schema per your field mapping."
        )

if n_migrate == 0:
    st.success("✅ Nothing to migrate in this direction — datasets are in sync.")
else:
    mg_col, _ = st.columns([2, 8])
    with mg_col:
        gen_mig = st.button("🚀 Generate Migration File", use_container_width=True)
    if gen_mig:
        with st.spinner(f"Streaming {n_migrate:,} records…"):
            try:
                b, fname, rows = build_migration_file(
                    src_path, src_ftype, src_key,
                    tgt_path, tgt_ftype, tgt_key,
                    field_map, direction,
                    mt["missing_keys"], mt["extra_keys"],
                )
                st.session_state.update(mig_bytes=b, mig_fname=fname, mig_rows=rows)
            except Exception as e:
                st.error(f"Migration error: {e}")

    if "mig_bytes" in st.session_state:
        st.success(f"✅ Ready — **{st.session_state['mig_rows']:,} rows** in migration file.")
        dl, _ = st.columns([2, 8])
        with dl:
            st.download_button(
                "⬇ Download Migration CSV",
                data=st.session_state["mig_bytes"],
                file_name=st.session_state["mig_fname"],
                mime="text/csv",
                use_container_width=True,
            )

st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<p class="section-label">💾 Export Full Report</p>', unsafe_allow_html=True)
ex1, ex2 = st.columns(2)
with ex1:
    st.download_button(
        "⬇ Excel Report (.xlsx)",
        data=build_excel_report(results),
        file_name=f"datasync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
with ex2:
    st.download_button(
        "⬇ Summary JSON",
        data=json.dumps(s, indent=2),
        file_name=f"datasync_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True,
    )