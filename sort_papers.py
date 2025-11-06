#!/usr/bin/env python3
"""
PDFSort — Offline PDF Topic Sorter (Embeddings + Clustering)

This version adds thorough comments to explain each step of the
pipeline: configuration, data loading, embedding, assignment, clustering,
and auto‑naming. Functionality is unchanged.

High‑level flow:
 1) Load topic seeds from seeds.yaml
 2) Embed seeds and PDFs (first N pages)
 3) Assign PDFs to the most similar seed if confidence ≥ THRESH_ASSIGN
 4) Send low‑confidence items to out/_unsorted
 5) If there are enough unsorted items, cluster them by cosine distance
 6) Auto‑name clusters (KeyBERT → TF‑IDF → word frequency) and move PDFs
 7) Write reports: assignment_report.csv, cluster_suggestions.json, _RUN_OK.json (implicit success marker)

All processing is local/offline — no external services are called.
"""

# ------------------------- STANDARD LIB -------------------------
# Built‑ins for filesystem ops, serialization, timing, text utils
import os, json, math, time, pathlib, shutil, re, unicodedata
from typing import List, Dict

# --------------------------- NUMPY/PANDAS -----------------------
# Numpy for vector math, Pandas to produce CSV reports
import numpy as np
import pandas as pd

# ----------------------------- PDF ------------------------------
# pypdf extracts text from PDF pages (no OCR)
from pypdf import PdfReader

# ---------------------- EMBEDDINGS/MODELS ----------------------
# Sentence‑Transformers for text embeddings (384‑dim for MiniLM)
from sentence_transformers import SentenceTransformer

# Agglomerative clustering to group low‑confidence PDFs by cosine
from sklearn.cluster import AgglomerativeClustering

# TF‑IDF used as a fallback for auto‑naming clusters
from sklearn.feature_extraction.text import TfidfVectorizer

# Pretty console output
from rich import print

# YAML for reading topic seeds and descriptions
import yaml

# Optional: KeyBERT for keyphrase extraction when naming clusters
from keybert import KeyBERT

# ========================= CONFIGURATION ========================
# You can tune these knobs to balance speed/quality and assignment behavior.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Small, fast, robust SBERT model (384‑d)
OUTDIR = "out"                           # Where outputs are written
PDFDIR = "pdfs"                          # Input directory of PDFs to sort
SEED_FILE = "seeds.yaml"                 # Topic definitions + short seed sentences

THRESH_ASSIGN = 0.30   # Confidence threshold to auto‑assign to a topic
                       # ↑ stricter → more files land in _unsorted
                       # ↓ looser   → more files auto‑assigned (risk of wrong bins)

MIN_TOKENS = 400       # Warn when a PDF yields very little text (often scanned)
MAX_PAGES_PER_DOC = 20 # Number of pages to read per PDF (↑ more context, slower)

# ====================== GLOBAL MODEL OBJECTS ====================
# Instantiate models once (global) so we don't re‑load per function call.
_embedder = SentenceTransformer(EMBED_MODEL_NAME)
_kw_model = KeyBERT(model=EMBED_MODEL_NAME)

# ========================= HELPER FUNCTIONS =====================

def embed(text: str) -> np.ndarray:
    """Return a normalized embedding vector for the given text.

    We request normalize_embeddings=True in encode() so vectors are
    unit‑length, which makes cosine similarity equivalent to dot product.
    """
    v = _embedder.encode([text], normalize_embeddings=True)[0]
    return np.array(v, dtype=np.float32)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors, with zero‑guard.

    Although the embedder already returns unit vectors, this function
    remains defensive against zero vectors.
    """
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_text(pdf_path: str, max_pages: int = MAX_PAGES_PER_DOC) -> str:
    """Extract text from the first `max_pages` pages of a PDF.

    pypdf is layout‑agnostic and returns raw text. For scanned PDFs you may
    get empty strings — this script does not run OCR.
    """
    try:
        reader = PdfReader(pdf_path)
        pages = min(len(reader.pages), max_pages)
        chunks = []
        for i in range(pages):
            s = reader.pages[i].extract_text() or ""
            chunks.append(s)
        return "\n".join(chunks)
    except Exception as e:
        print(f"[yellow]Warn:[/yellow] {pdf_path} -> {e}")
        return ""


def ensure_dir(p: str) -> None:
    """Create a folder (and parents) if it does not already exist."""
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


# -------- YAML LOADER (robust to BOM and “smart quotes”) --------

def load_yaml_utf8(path: str) -> dict:
    """Load YAML with UTF‑8 handling and punctuation normalization.

    Addresses common Windows issues (BOM) and replaces curly quotes/dashes
    with ASCII equivalents so topic names and seeds are consistent.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        # Fall back to utf‑8‑sig and replace undecodable bytes
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            text = f.read()
    # Normalize a few common Unicode punctuation cases
    text = (text
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2013", "-").replace("\u2014", "-")
            .replace("\xa0", " "))
    # NFC for consistent codepoint composition
    text = unicodedata.normalize("NFC", text)
    return yaml.safe_load(text)


# --------------- CLUSTER TITLE SAMPLING / SUMMARIES -------------

def summarize_cluster_titles(files: List[str], base_dir: str, max_pages: int = 3):
    """Collect quick title/snippet pairs from a handful of files.

    Used by the auto‑naming pipeline to build a small text blob that
    represents a cluster's content.
    """
    texts = []
    titles = []
    for fn in files[:6]:  # keep it small and fast
        path = os.path.join(base_dir, fn)
        # Title hint from filename, replacing underscores/dashes
        title = pathlib.Path(fn).stem.replace("_", " ").replace("-", " ")
        # Extract a tiny snippet
        snippet = extract_text(path, max_pages=max_pages)
        snippet = re.sub(r"\s+", " ", (snippet or ""))[:2000]
        titles.append(title)
        texts.append(f"{title}. {snippet}")
    return titles, " \n".join(texts)


# ---------------------- AUTO‑NAMING PIPELINE --------------------

def propose_cluster_name(files: List[str], base_unsorted: str) -> str:
    """Propose a short, human‑readable name for a cluster of PDFs.

    Strategy (ordered):
      1) KeyBERT keyphrases over titles+snippets
      2) TF‑IDF of titles
      3) Fallback: simple word frequency of titles

    The chosen phrase is sanitized for Windows‑safe folder names.
    """
    # Build a small text blob from a sample of the cluster
    titles, blob = summarize_cluster_titles(files, base_unsorted)
    candidates: List[str] = []

    # 1) Try KeyBERT (embedding‑aware)
    try:
        phrases = _kw_model.extract_keywords(
            blob,
            keyphrase_ngram_range=(1, 3),
            stop_words='english',
            use_mmr=True,
            diversity=0.6,
            top_n=5
        )
        candidates = [p[0] for p in phrases if len(p[0]) >= 3]
    except Exception:
        # If KeyBERT fails (e.g., empty text), continue to fallbacks
        pass

    # 2) TF‑IDF fallback on titles
    if not candidates and titles:
        try:
            tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            X = tf.fit_transform(titles)
            means = np.asarray(X.mean(axis=0)).ravel()
            idx = means.argsort()[::-1][:3]
            feats = np.array(tf.get_feature_names_out())[idx]
            cand = " ".join(feats).strip()
            if cand:
                candidates = [cand]
        except Exception:
            pass

    # 3) Final fallback: most frequent non‑trivial words
    if not candidates:
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", " ".join(titles).lower())
        stop = {
            "the","and","for","with","from","into","using","data","based",
            "study","system","model","approach","method","analysis","results"
        }
        freq: Dict[str, int] = {}
        for w in words:
            if w in stop:
                continue
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
        candidates = [" ".join([w for w, _ in top]) or "Cluster"]

    # Sanitize to a tidy folder‑ready title
    name = candidates[0].title()
    name = re.sub(r"[^\w \-]", "", name).strip()  # drop punctuation
    name = name[:80] or "Cluster"                 # length cap & default
    return name


# ---------------- AGGLOMERATIVE (COSINE) WRAPPER ----------------

def agglomerative_cosine(X: np.ndarray, n_clusters: int) -> np.ndarray:
    """Run average‑linkage agglomerative clustering with cosine metric.

    scikit‑learn changed the param name from `affinity`→`metric` in recent
    versions; this wrapper keeps compatibility with older installs.
    """
    try:
        cl = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    except TypeError:
        cl = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="average")
    labels = cl.fit_predict(X)
    return labels


# ============================== MAIN ============================

def main() -> None:
    """Entrypoint: orchestrates sorting, clustering, and reporting."""
    # Ensure output structure exists
    ensure_dir(OUTDIR)
    ensure_dir(os.path.join(OUTDIR, "_unsorted"))

    # ---- Load topics/seeds ----
    seeds = load_yaml_utf8(SEED_FILE)
    topics = seeds["topics"]
    # If a topic is missing a `seed`, fall back to using its `name`
    for t in topics:
        if not t.get("seed"):
            t["seed"] = t["name"]

    # ---- Embed topic seeds ----
    for t in topics:
        t["vec"] = embed(t["seed"])  # precompute once; reused for all PDFs

    # ---- Iterate PDFs, embed, and score against seeds ----
    rows = []                 # rows for assignment_report.csv
    pdf_files = sorted([f for f in os.listdir(PDFDIR) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print("[yellow]No PDFs found in 'pdfs/' — add files and re-run.[/yellow]")

    doc_vecs: List[np.ndarray] = []  # cache vectors to avoid recomputing during clustering
    doc_names: List[str] = []        # parallel list of filenames

    for fn in pdf_files:
        path = os.path.join(PDFDIR, fn)
        text = extract_text(path)
        # Token budget warning — helps diagnose scanned or very short docs
        if len((text or "").split()) < MIN_TOKENS:
            print(f"[yellow]Low text[/yellow] ({len((text or '').split())} tokens): {fn}")

        # Provide a title hint from filename to the embedder for extra signal
        title_hint = pathlib.Path(fn).stem.replace("_", " ").replace("-", " ")
        doc_text = f"Title: {title_hint}\n\n{text[:40000]}"  # cap to keep memory/runtimes bounded
        v = embed(doc_text)
        doc_vecs.append(v)
        doc_names.append(fn)

        # Compute cosine similarity to each topic seed and choose best
        sims = [(t["name"], cos(v, t["vec"])) for t in topics]
        sims.sort(key=lambda x: x[1], reverse=True)
        best_topic, best_sim = sims[0]

        rows.append({
            "file": fn,
            "best_topic": best_topic,
            "best_sim": round(best_sim, 3),
            "top3": json.dumps([(n, round(s, 3)) for n, s in sims[:3]])
        })

    # ---- Save the assignment report (preview of decisions) ----
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "assignment_report.csv"), index=False)

    # ---- Materialize topic folders and copy files by confidence ----
    topic_to_dir: Dict[str, str] = {}
    for t in topics:
        d = os.path.join(OUTDIR, t["name"])  # folder name is the topic name
        ensure_dir(d)
        topic_to_dir[t["name"]] = d

    for _, r in df.iterrows():
        src = os.path.join(PDFDIR, r["file"])  # original location
        if r["best_sim"] >= THRESH_ASSIGN:
            # Meets threshold ⇒ copy into its winning topic folder
            dst = os.path.join(topic_to_dir[r["best_topic"]], r["file"])
        else:
            # Below threshold ⇒ place into the low‑confidence bin
            dst = os.path.join(OUTDIR, "_unsorted", r["file"])
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"[red]Copy fail:[/red] {src} -> {dst}: {e}")

    # ---- Cluster low‑confidence PDFs to suggest subtopics ----
    unsorted = [f for f in os.listdir(os.path.join(OUTDIR, "_unsorted")) if f.lower().endswith(".pdf")]
    cluster_map: Dict[int, List[str]] = {}

    # Clustering only makes sense with a small batch; require ≥ 3
    if len(unsorted) >= 3:
        X = []
        names = []
        for fn in unsorted:
            # Reuse previously computed vectors when possible
            idx = doc_names.index(fn) if fn in doc_names else None
            if idx is None:
                # Rare path: if the file name changed or cache missed, re‑embed quickly
                text = extract_text(os.path.join(OUTDIR, "_unsorted", fn))
                v = embed(text[:40000])
            else:
                v = doc_vecs[idx]
            X.append(v)
            names.append(fn)
        X = np.vstack(X)

        try:
            # Heuristic cluster count: between 2 and 5 depending on pile size
            n_clusters = min(5, max(2, len(unsorted) // 5))
            labels = agglomerative_cosine(X, n_clusters=n_clusters)
            for name, lab in zip(names, labels):
                cluster_map.setdefault(int(lab), []).append(name)
        except Exception as e:
            print(f"[yellow]Clustering skipped:[/yellow] {e}")

    # Persist raw cluster suggestions for transparency/debugging
    with open(os.path.join(OUTDIR, "cluster_suggestions.json"), "w") as f:
        json.dump(cluster_map, f, indent=2)

    # ---- Auto‑name clusters and move into _unsorted_named/ ----
    if cluster_map:
        base_unsorted = os.path.join(OUTDIR, "_unsorted")
        named_root = os.path.join(OUTDIR, "_unsorted_named")
        ensure_dir(named_root)
        for cid, files in cluster_map.items():
            # Try to produce a short descriptive folder name for each cluster
            folder = propose_cluster_name(files, base_unsorted)
            target_dir = os.path.join(named_root, folder)
            ensure_dir(target_dir)
            for fn in files:
                src = os.path.join(base_unsorted, fn)
                dst = os.path.join(target_dir, fn)
                try:
                    shutil.move(src, dst)
                except Exception as e:
                    print(f"[yellow]Move failed:[/yellow] {src} -> {dst}: {e}")
        print("[green]Auto-named subgroups created under out/_unsorted_named (offline).[/green]")

    print("[green]Done.[/green] Outputs in 'out/'.")


# --------------------------- CLI GUARD --------------------------
if __name__ == "__main__":
    main()
