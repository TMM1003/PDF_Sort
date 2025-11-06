#!/usr/bin/env python3
"""
LEARNING NOTES:
- This script assigns PDFs to topics using semantic similarity to short seed sentences.
- If a PDF falls below the similarity threshold, it is placed into out/_unsorted/.
- When there are enough unsorted PDFs (>=3), they are clustered to propose sub‑groups; clusters are auto‑named.
- Adjust THRESH_ASSIGN / MAX_PAGES_PER_DOC / seeds.yaml to tune behavior.

Paper sorter (offline): embeddings + semi-supervised assignment + clustering + auto-naming (offline (no external services))
- Embeddings: sentence-transformers (default: all-MiniLM-L6-v2)
- PDF text extraction: pypdf
- Clustering: Agglomerative (cosine)
- Auto-naming: KeyBERT (keyphrase extraction) with TF-IDF fallback
"""

import os, json, math, time, pathlib, shutil, re, unicodedata
from typing import List, Dict

import numpy as np
import pandas as pd
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from rich import print
import yaml

# Optional naming helper
from keybert import KeyBERT

# ---------------------- CONFIG ----------------------
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # Embedding backbone: fast, robust, offline
# You can switch models (e.g., "all-mpnet-base-v2") for quality vs speed tradeoffs.
OUTDIR = "out"
PDFDIR = "pdfs"
SEED_FILE = "seeds.yaml"

THRESH_ASSIGN = 0.30   # Similarity threshold to auto-assign (↑ stricter = more in _unsorted; ↓ looser = more auto-assigned)
MIN_TOKENS = 400       # Warn if extracted text is very short (often scanned PDFs). Adjust as needed.
MAX_PAGES_PER_DOC = 20 # How many pages to read per PDF (↑ captures more context but takes longer).

# ------------------- INITIALIZATION -----------------
_embedder = SentenceTransformer(EMBED_MODEL_NAME)
_kw_model = KeyBERT(model=EMBED_MODEL_NAME)

def embed(text: str) -> np.ndarray:
    v = _embedder.encode([text], normalize_embeddings=True)[0]
    return np.array(v, dtype=np.float32)

def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def extract_text(pdf_path: str, max_pages=MAX_PAGES_PER_DOC) -> str:
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

def ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# Robust UTF-8 YAML loader (tolerates BOM and smart punctuation)
def load_yaml_utf8(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
            text = f.read()
    text = (text
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2013", "-").replace("\u2014", "-")
            .replace("\xa0", " "))
    text = unicodedata.normalize("NFC", text)
    return yaml.safe_load(text)

def summarize_cluster_titles(files, base_dir, max_pages=3):
    texts = []
    titles = []
    for fn in files[:6]:  # small sample
        path = os.path.join(base_dir, fn)
        title = pathlib.Path(fn).stem.replace("_"," ").replace("-"," ")
        snippet = extract_text(path, max_pages=max_pages)
        snippet = re.sub(r"\s+", " ", (snippet or ""))[:2000]
        titles.append(title)
        texts.append(f"{title}. {snippet}")
    return titles, " \n".join(texts)

def propose_cluster_name(files, base_unsorted):
    # 1) KeyBERT keyphrases
    titles, blob = summarize_cluster_titles(files, base_unsorted)
    candidates = []
    try:
        phrases = _kw_model.extract_keywords(
            blob,
            keyphrase_ngram_range=(1,3),
            stop_words='english',
            use_mmr=True,
            diversity=0.6,
            top_n=5
        )
        candidates = [p[0] for p in phrases if len(p[0]) >= 3]
    except Exception as e:
        pass

    # 2) Fallback: TF-IDF on titles
    if not candidates:
        if titles:
            try:
                tf = TfidfVectorizer(stop_words="english", ngram_range=(1,2), min_df=1)
                X = tf.fit_transform(titles)
                # pick top features by mean tf-idf
                means = np.asarray(X.mean(axis=0)).ravel()
                idx = means.argsort()[::-1][:3]
                feats = np.array(tf.get_feature_names_out())[idx]
                cand = " ".join(feats).strip()
                if cand:
                    candidates = [cand]
            except Exception:
                pass

    # 3) Fallback: frequency of words
    if not candidates:
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", " ".join(titles).lower())
        stop = {"the","and","for","with","from","into","using","data","based","study",
                "system","model","approach","method","analysis","results"}
        freq = {}
        for w in words:
            if w in stop: continue
            freq[w] = freq.get(w, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
        candidates = [" ".join([w for w,_ in top]) or "Cluster"]

    name = candidates[0].title()
    name = re.sub(r"[^\w \-]", "", name).strip()
    name = name[:80] or "Cluster"
    return name

def agglomerative_cosine(X, n_clusters):
    # sklearn 1.4+ uses 'metric', older uses 'affinity'
    try:
        cl = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    except TypeError:
        cl = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="average")
    labels = cl.fit_predict(X)
    return labels

def main():
    ensure_dir(OUTDIR)
    ensure_dir(os.path.join(OUTDIR, "_unsorted"))

    seeds = load_yaml_utf8(SEED_FILE)
    topics = seeds["topics"]
    for t in topics:
        if not t.get("seed"):
            t["seed"] = t["name"]
    # 1) Embed topic seeds
    for t in topics:
        t["vec"] = embed(t["seed"])

    # 2) Process PDFs
    rows = []
    pdf_files = sorted([f for f in os.listdir(PDFDIR) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print("[yellow]No PDFs found in 'pdfs/' — add files and re-run.[/yellow]")
    doc_vecs, doc_names = [], []

    for fn in pdf_files:
        path = os.path.join(PDFDIR, fn)
        text = extract_text(path)
        if len((text or "").split()) < MIN_TOKENS:
            print(f"[yellow]Low text[/yellow] ({len((text or '').split())} tokens): {fn}")
        title_hint = pathlib.Path(fn).stem.replace("_"," ").replace("-"," ")
        doc_text = f"Title: {title_hint}\n\n{text[:40000]}"
        v = embed(doc_text)
        doc_vecs.append(v); doc_names.append(fn)

        sims = [(t["name"], cos(v, t["vec"])) for t in topics]  # cosine similarity to each topic seed
        sims.sort(key=lambda x: x[1], reverse=True)
        best_topic, best_sim = sims[0]

        rows.append({
            "file": fn,
            "best_topic": best_topic,
            "best_sim": round(best_sim, 3),
            "top3": json.dumps([(n, round(s,3)) for n,s in sims[:3]])
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "assignment_report.csv"), index=False)

    # 3) Materialize folders by confidence
    topic_to_dir = {}
    for t in topics:
        d = os.path.join(OUTDIR, t["name"])
        ensure_dir(d)
        topic_to_dir[t["name"]] = d

    for _, r in df.iterrows():
        src = os.path.join(PDFDIR, r["file"])
        if r["best_sim"] >= THRESH_ASSIGN:  # meets confidence threshold → copy into the winning topic
            dst = os.path.join(topic_to_dir[r["best_topic"]], r["file"])
        else:
            # below threshold → send to the unsorted bin (candidate for clustering)
            dst = os.path.join(OUTDIR, "_unsorted", r["file"])
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            print(f"[red]Copy fail:[/red] {src} -> {dst}: {e}")

    # 4) Cluster the unsorted set to propose subtopics
    unsorted = [f for f in os.listdir(os.path.join(OUTDIR, "_unsorted")) if f.lower().endswith(".pdf")]
    cluster_map = {}
    # Only cluster when there are at least 3 unsorted PDFs; otherwise cluster_suggestions.json will be empty ({{}})
    if len(unsorted) >= 3:
        X = []
        names = []
        for fn in unsorted:
            idx = doc_names.index(fn) if fn in doc_names else None
            if idx is None: 
                # embed quickly from file again (rare path)
                text = extract_text(os.path.join(OUTDIR, "_unsorted", fn))
                v = embed(text[:40000])
            else:
                v = doc_vecs[idx]
            X.append(v)
            names.append(fn)
        X = np.vstack(X)
        try:
            n_clusters = min(5, max(2, len(unsorted)//5))
            labels = agglomerative_cosine(X, n_clusters=n_clusters)
            for name, lab in zip(names, labels):
                cluster_map.setdefault(int(lab), []).append(name)
        except Exception as e:
            print(f"[yellow]Clustering skipped:[/yellow] {e}")

    with open(os.path.join(OUTDIR, "cluster_suggestions.json"), "w") as f:
        json.dump(cluster_map, f, indent=2)

    # 5) Auto-name clusters & move into _unsorted_named
    if cluster_map:
        base_unsorted = os.path.join(OUTDIR, "_unsorted")
        named_root = os.path.join(OUTDIR, "_unsorted_named")
        ensure_dir(named_root)
        for cid, files in cluster_map.items():
            folder = propose_cluster_name(files, base_unsorted)  # KeyBERT → TF‑IDF → word frequency fallback
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

if __name__ == "__main__":
    main()
