# PDFSort — Offline Semantic PDF Organization Tool

PDFSort is an **offline, machine-learning–based PDF organization tool** that uses
semantic text embeddings and unsupervised clustering to automatically group and
label documents by topic.

Unlike rule-based filename sorters, PDFSort analyzes the *content* of PDF files
using transformer-based embeddings, enabling semantic classification and discovery
of previously unknown document categories.

All processing is performed locally — **no external APIs or cloud services are used**.

---

## Project Overview

PDFSort is designed for organizing large collections of PDF documents such as
academic papers, reports, and technical documentation.

The system combines:
- **Semantic embeddings** for content understanding
- **Similarity-based assignment** for known topics
- **Unsupervised clustering** for unknown or ambiguous documents
- **Automated cluster naming** using NLP techniques

The result is a transparent, reproducible document-sorting pipeline that balances
automation with interpretability.

---

## Core Capabilities

- Transformer-based **semantic embeddings** of PDF content
- Topic assignment using **cosine similarity with confidence thresholds**
- **Unsupervised clustering** of low-confidence documents
- Automatic, human-readable **cluster naming**
- Fully **offline execution**
- Detailed CSV and JSON reports for auditing and analysis
- Safe, non-destructive file operations

---

## How It Works

### 1. Topic Seeding
Topics and short seed descriptions are defined in a `seeds.yaml` file.
Each seed is embedded and used as a semantic anchor for classification.

### 2. PDF Text Extraction
The tool extracts text from the first *N* pages of each PDF (configurable)
using `pypdf`. No OCR is performed.

### 3. Semantic Embedding
Document text and topic seeds are embedded using a
**Sentence-Transformer (MiniLM)** model.
Embeddings are normalized to support cosine similarity.

### 4. Similarity-Based Assignment
Each document is compared against all topic seeds:
- Documents with similarity above a configurable threshold are assigned directly
- Low-confidence documents are routed to an `_unsorted` pool

### 5. Unsupervised Clustering
If enough documents remain unassigned, the tool applies
**agglomerative clustering (cosine distance)** to group them into latent topics.

### 6. Automatic Cluster Naming
Clusters are labeled using a multi-stage NLP pipeline:
1. **KeyBERT** keyphrase extraction
2. **TF-IDF** fallback on document titles
3. Word-frequency fallback as a last resort

### 7. Reporting
The system generates:
- `assignment_report.csv` — similarity scores and topic decisions
- `cluster_suggestions.json` — raw clustering output
- `_RUN_OK.json` — implicit success marker

---

## Example Use Cases

- Organizing academic research papers by topic
- Sorting course materials or technical documentation
- Cleaning large PDF download folders
- Discovering latent themes in document collections

---

## Configuration

Key parameters can be tuned to control behavior:

- Embedding model selection
- Assignment confidence threshold
- Maximum pages extracted per PDF
- Minimum document size warnings
- Clustering granularity

These allow trade-offs between speed, precision, and recall.

---

## System Requirements

- Python 3.8+
- Runs on Windows, macOS, and Linux
- All processing is local/offline

### Key Dependencies

- `sentence-transformers`
- `scikit-learn`
- `pypdf`
- `keybert`
- `numpy`, `pandas`, `yaml`

---

## Design Philosophy

PDFSort prioritizes:
- **Interpretability** over black-box automation
- **Offline reproducibility**
- **Minimal assumptions about document structure**
- **Clear separation between assignment and discovery**

This makes it suitable both as a practical utility and as an applied NLP / ML project.

---

## Usage

```bash
python sort_papers.py
```
## Note
- Scanned PDFs without extractable text will yield poor embeddings
- This tool does not perform OCR
- Topic quality depends on seed descriptions and threshold tuning