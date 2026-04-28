# ⚡ DataSync — Dataset Reconciliation Tool

A sleek Streamlit app for comparing two datasets (CSV / Excel) and surfacing mismatches, duplicates, nulls, and field-level differences — with an optional AI-generated findings summary powered by Claude (Anthropic).

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🧰 Features

| Feature | Details |
|---|---|
| **File ingestion** | CSV and Excel (.xlsx / .xls) |
| **Key mapping** | Map source key column → target key column |
| **Field mapping** | Map N source fields → N target fields for value comparison |
| **Missing records** | Records in source not found in target |
| **Extra records** | Records in target not found in source |
| **Duplicate detection** | Duplicate keys in source and/or target |
| **Field mismatches** | Same key, different field values |
| **Null / blank values** | Null or empty string detection across all columns |
| **Summary metrics** | Match rate %, total errors, per-issue counts |
| **AI summary** | Prose narrative using Claude (Anthropic API) |
| **Excel export** | Multi-sheet report with all issue categories |
| **JSON export** | Machine-readable summary metrics |

---

## 🔑 API Key (for AI Summary)

Enter your **Openai API key** in the sidebar (`sk-ant-...`).  
Get one at: https://openai.com/api/

The AI summary is optional — all comparison features work without it.

---

## 📁 Sample Data

Use the included sample files to test the tool:

- `sample_source.csv` — 14 rows with a duplicate (E002)
- `sample_target.csv` — 14 rows with intentional mismatches:
  - **E002**: salary changed 72000 → 75000
  - **E003**: department changed Engineering → Product
  - **E004**: status changed Active → Inactive
  - **E010**: salary changed 91000 → 94000
  - **E007**: missing email (null)
  - **E012, E013**: missing in target
  - **E014, E015**: extra in target
  - **E009**: duplicated in target

**Suggested field mapping (source → target):**
- `salary` → `annual_salary`
- `department` → `dept`
- `status` → `employment_status`
- `email` → `work_email`

---

## 🏗️ Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **Pandas** — Data comparison engine
- **OpenPyXL** — Excel export
- **Anthropic SDK** — AI findings summary (Claude)
