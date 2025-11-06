# NormSort: Offline PDF Topic Sorter (Embeddings + Clustering)

NormSort sorts research PDFs into topic folders using **local** sentence embeddings (no internet, no Ollama).
It can also cluster any low-confidence items and **auto-name subgroups** (KeyBERT/TF‑IDF).

> Works on Windows, Tested on Windows with VS Code.
>
> Unconfirmed: macOS, and Linux.

---

## ✨ Features

- **Semantic sorting** with SentenceTransformers embeddings (default: `all-MiniLM-L6-v2`)
- **Seeded topics** via `seeds.yaml` (semi‑supervised)
- **Clustering** of low-confidence papers
- **Auto‑naming** of subgroups (KeyBERT + TF‑IDF, fully offline)
- **Run confirmation** file: `out/_RUN_OK.json` with counts & timestamp
- Windows‑safe folder names (slashes and other illegal characters are sanitized)

---

## 🧰 Requirements

- Python **3.9+** recommended
- Install dependencies:

```bash
 python -m pip install -r requirements.txt
```

> If you use a virtual environment, activate it before installing.

---

## 📦 Project Layout

```
PDF_Sort/ # your project root (any name)
├─ pdfs/ # put your PDFs here
├─ seeds.yaml # topic names + 1–2 sentence seed blurbs
├─ sort_papers.py # the sorter
├─ requirements.txt
└─ out/ # created on first run
 ├─ <Topic Name>/*.pdf
 ├─ _unsorted/*.pdf
 ├─ _unsorted_named/<auto-subtopic>/*.pdf
 ├─ assignment_report.csv
 ├─ cluster_suggestions.json
 └─ _RUN_OK.json # success marker + summary
```

> On Windows, keep `pdfs/`, `seeds.yaml`, and `sort_papers.py` in the **same folder** (e.g., `C:\Projects\PDF_Sort\`).

---

## 🚀 Quickstart

1. **Open the folder** in VS Code (recommended) or a terminal:

- VS Code → _File → Open Folder…_
- Or CMD/PowerShell: `cd C:\Projects\PDF_Sort`

2. **Install packages**:

```bash
 python -m pip install -r requirements.txt
```

3. **Add PDFs** to the `pdfs/` directory.
4. **Edit topics** in `seeds.yaml` (UTF‑8). Keep **slashes out of `name`** values on Windows.
5. **Run the sorter**:

```bash
 python sort_papers.py
```

6. **Check outputs** in `out/`. For a quick success check, open `out/_RUN_OK.json`.

---

## ⚙️ Configuration (edit inside `sort_papers.py`)

```python
EMBED_MODEL_NAME = "all-MiniLM-L6-v2" # small, fast, offline
OUTDIR = "out"
PDFDIR = "pdfs"
SEED_FILE = "seeds.yaml"

THRESH_ASSIGN = 0.30 # ↑ for stricter auto-assign (e.g., 0.35–0.45)
MIN_TOKENS = 400 # warn if too little text extracted
MAX_PAGES_PER_DOC = 20 # ↑ for more context; slower
```

- **THRESH_ASSIGN**: if similarity < threshold → goes to `_unsorted`.
- **MAX_PAGES_PER_DOC**: more pages = better context, higher runtime.

---

## 🗂 About `seeds.yaml`

- `name`: Human‑readable topic label (becomes a **folder name**). Avoid Windows‑illegal characters: `\ / : * ? " < > |` and trailing dots/spaces.
- `seed`: A 1–2 sentence description used for semantic anchoring. It **can** contain slashes; it is not used as a folder name.

Example:

```yaml
topics:
 - name: "On-device - Edge & Split Computing"
 seed: "Real-time ML on device; edge offloading; latency and energy constraints."
 - name: "BCI & Robotic–Prosthetic Control"
 seed: "BCI and myoelectric prostheses; intent decoding; closed-loop safety."
```

---

## 📊 Outputs

- **Topic folders** under `out/<Topic Name>/`
- **Low-confidence bin**: `out/_unsorted/`
- **Auto-named subgroups** (from clusters): `out/_unsorted_named/<short-name>/`
- **Report CSV**: `out/assignment_report.csv` (file → best topic, confidence, top‑3)
- **Clusters JSON**: `out/cluster_suggestions.json`
- **Run confirmation**: `out/_RUN_OK.json`

---

## 🧪 Verifying a Successful Run

- Human check: open `out/_RUN_OK.json` and review counts.
- Scriptable check (PowerShell):

```powershell
 python .\sort_papers.py
 if (Test-Path .\out\_RUN_OK.json) { "OK"; Get-Content .\out\_RUN_OK.json } else { "FAILED" }
```

---

## 🔧 Tuning Tips

- Raise `THRESH_ASSIGN` to reduce false positives.
- Add more precise `seed` blurbs (short, specific sentences).
- Increase `MAX_PAGES_PER_DOC` for dense/technical PDFs.
- If many PDFs are **scans** (image‑only), consider adding OCR later (e.g., Tesseract) for better text extraction.

---

## 🩹 Troubleshooting

**Unicode error loading `seeds.yaml`**

- Ensure the file is UTF‑8. In VS Code: bottom‑right encoding → _Reopen with Encoding → UTF‑8_; then _Save with Encoding → UTF‑8_.
- The script already opens YAML with `encoding="utf-8"` and normalizes “smart” punctuation.

**Windows folder name errors**

- Keep slashes out of `name`. The script sanitizes names (`safe_folder_name`), but warnings mean you should fix the YAML too.

**`AgglomerativeClustering` complains about `metric`**

- Old scikit‑learn: code auto‑falls back to `affinity="cosine"`.

**Nothing gets assigned**

- Lower the threshold, improve seed blurbs, or increase pages per doc. Some scans may require OCR.

---

## 🤖 How it Works (High Level)

1. Read & embed each **topic seed** and **paper** (first _N_ pages).
2. Assign paper → topic with highest cosine similarity (if ≥ threshold).
3. Put low-confidence items into `_unsorted`.
4. Cluster the unsorted pile (agglomerative, cosine).
5. Auto‑name each cluster using **KeyBERT** keyphrases (fallback: TF‑IDF / word frequency).
6. Move clustered PDFs into `out/_unsorted_named/<auto-name>/`.
7. Write `out/_RUN_OK.json` summary.

---

## ❓ Q & A

### Q1) What is `all-MiniLM-L6-v2`?

A small, fast **SentenceTransformers** model (SBERT family) that maps text to **384‑dimensional embeddings** capturing **semantic meaning** (not just keywords). “MiniLM” is the distilled Transformer backbone; **L6** means 6 layers; **v2** is the improved release.

### Q2) How does it find commonalities without a keyword list?

It compares **embeddings**:

- Each paper → one embedding (from its extracted text).
- Each topic seed → one embedding.
- It computes **cosine similarity** and assigns a paper to the closest topic if the score ≥ `THRESH_ASSIGN`.
  Papers below the threshold go to `_unsorted`, get **clustered** by cosine similarity, and clusters are **auto‑named** via KeyBERT/TF‑IDF.

### Q3) Where do the “common words” come from for auto‑naming?

From each cluster’s **titles/snippets**:

- **KeyBERT** proposes representative phrases (embedding‑aware, reduces redundancy via MMR).
- If that fails, **TF‑IDF** surfaces salient n‑grams; final fallback is word frequency on titles.

### Q4) Can I run entirely offline?

Yes. All models used (`sentence-transformers`, KeyBERT) run locally. No internet or Ollama is required.

### Q5) How do I change models?

Edit in `sort_papers.py`:

```python
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
```

Good alternatives: `"all-mpnet-base-v2"` (higher quality, slower), `"bge-small-en-v1.5"` (strong small model). Re‑run after changing.

### Q6) Why did some PDFs land in `_unsorted`?

Their best similarity was **below** `THRESH_ASSIGN`, or the PDF had **too little text** extracted (e.g., scanned images). Try raising `MAX_PAGES_PER_DOC`, improving seed blurbs, or adding OCR later.

### Q7) How do I tune assignment behavior?

Increase `THRESH_ASSIGN` to be stricter (fewer autos; more in `_unsorted`), or decrease for broader assignment. Adjust pages and seed specificity to improve accuracy.

### Q8) How are folder names kept Windows‑safe?

All topic and subgroup names pass through `safe_folder_name()` which replaces illegal characters (`\ / : * ? " < > |`), collapses whitespace, and trims trailing dots/spaces.

### Q9) How do I verify a successful run?

Check for `out/_RUN_OK.json`. It includes timestamp, counts per topic, and number of clusters created.

### Q10) Does GPU help?

SentenceTransformers can use a GPU if PyTorch sees one. For 40 PDFs, CPU is fine; GPU helps for very large batches.

### Q11) My scikit‑learn errors on `metric="cosine"`

Older versions use `affinity="cosine"`. The code already falls back automatically.

### Q12) Any privacy concerns?

Everything runs locally; no data leaves your machine.

---

---

## 🧠 Learn the Software (How it Decides Things)

This section explains the pipeline in plain language so you can answer questions about how it works.

### 1) Topic Seeds = “Semantic Magnets”

- Each entry in `seeds.yaml` has a **name** (becomes a folder name) and a short **seed** sentence.
- The seed sentence is embedded (turned into a vector). Think of it like a **magnet** that pulls in semantically similar papers.
- Keep names Windows‑safe (no `\ / : * ? " < > |`)—the script sanitizes names anyway, but it’s best to avoid those in `name`.

**Editing tips:**

- Keep seeds **short (1–2 sentences)** and **specific**. Include distinctive terms to separate overlapping topics.
- You can adjust seeds any time and re‑run—no retraining involved.

### 2) Assignment Logic (When does a paper enter a topic folder?)

- The script extracts text from each PDF (up to `MAX_PAGES_PER_DOC` pages) and embeds it.
- It computes **cosine similarity** between the paper and every topic seed.
- If the **best** similarity ≥ `THRESH_ASSIGN`, the PDF is **copied into that topic’s folder**.
- If **below** the threshold, it goes to **`out/_unsorted/`**.

**You control this behavior with:**

- `THRESH_ASSIGN`: raise to be stricter (fewer false positives), lower to be more inclusive.
- `MAX_PAGES_PER_DOC`: increase if key details tend to appear later in your documents.
- `MIN_TOKENS`: warns when too little text was extracted (often scans).

### 3) Unsorted Criteria (Why did a PDF land in `_unsorted`?)

A PDF goes to `out/_unsorted/` if:

- Its best similarity was **below** `THRESH_ASSIGN`.
- Or it didn’t extract enough text to judge confidently (see token warnings).
- Or the seed blurbs weren’t specific enough for that paper’s content.

**What to try:**

- Strengthen the corresponding seed sentence.
- Increase `MAX_PAGES_PER_DOC`.
- Raise/lower `THRESH_ASSIGN` temporarily to observe behavior.

### 4) Cluster Suggestions (When is `cluster_suggestions.json` empty?)

- The script clusters **only when there are at least 3 PDFs** in `_unsorted/` by default.
- With fewer than 3, it writes an **empty `{}`** to `out/cluster_suggestions.json`.
- When clustering runs, the file maps cluster IDs → lists of filenames.

### 5) Auto‑Naming Subgroups (How are names chosen?)

- For each cluster, the script samples titles/snippets from a few PDFs and tries to extract **keyphrases**.
- It first uses **KeyBERT**; if that isn’t confident, it falls back to **TF‑IDF**; then to simple **word frequency**.
- The picked phrase is sanitized into a Windows‑safe folder and used under `out/_unsorted_named/<auto-name>/`.

### 6) Run Confirmation (How do I know it worked?)

- After a successful run, the script writes `out/_RUN_OK.json` with:
  - `total_pdfs`, `assigned`, `unsorted`
  - counts per topic
  - `clusters_created` (if any)

Open that file to quickly see the high‑level results.

## 🛠 Automation (optional)

**Windows Task Scheduler / batch**:

```bat
@echo off
cd /d C:\Projects\PDF_Sort
python -m pip install -r requirements.txt
python sort_papers.py
if exist out\_RUN_OK.json (echo OK) else (echo FAILED & exit /b 1)
```

---

## 🙌 Acknowledgements

- [SentenceTransformers](https://www.sbert.net/)
- [KeyBERT](https://github.com/MaartenGr/KeyBERT)
- [scikit‑learn](https://scikit-learn.org/)
- [PyPDF](https://pypdf.readthedocs.io/)
