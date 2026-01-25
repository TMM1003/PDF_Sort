# PDF Sort Utility

A lightweight **Python utility for automatically organizing PDF files** based on filename patterns and configurable rules.  
Designed to simplify document management workflows by programmatically sorting PDFs into structured directories.

---

## Project Overview

This project provides a simple, extensible solution for organizing PDF files without relying on external document-management tools. It scans a target directory, analyzes PDF filenames, and relocates files into categorized subfolders based on predefined sorting logic.

The primary focus of the project is **automation, correctness, and filesystem safety**, rather than user interface or third-party integrations.

---

## Core Features

- Automated discovery of PDF files within a target directory
- Rule-based sorting using filename patterns
- Automatic creation of destination folders
- Safe file movement with overwrite protection
- Minimal dependencies (Python standard library only)

---

## How It Works

1. **Scan Input Directory**  
   The script identifies all `.pdf` files in the configured source directory.

2. **Classify Files**  
   Each PDF is evaluated against a set of filename-based rules to determine its category.

3. **Organize Output**  
   Files are moved into category-specific subdirectories, which are created automatically if they do not already exist.

4. **Preserve File Integrity**  
   The script avoids destructive operations and ensures files are not silently overwritten.

---

## Example Use Cases

This utility is well-suited for organizing collections such as:

- Academic papers
- Course handouts
- Reports and invoices
- Download folders that accumulate PDFs over time

The sorting logic can be easily adapted to new naming conventions or folder structures.

---

## System Requirements

- Python 3.8 or newer
- Compatible with Windows, macOS, and Linux
- No external dependencies required

---

## Usage

### Run the Script

```bash
python pdf_sort.py
