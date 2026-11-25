#!/usr/bin/env python3
"""Local TF-IDF search over a folder of documents (no API needed).

Builds a TF-IDF index from plain-text files in a folder, then ranks
them by cosine similarity to a query. Standard library only.
"""
import argparse
import math
import os
import re
import sys
from collections import Counter

STOPWORDS = set("""a an the and or but if then else when at by for with about
against between into through during before after above below to from up down
in out on off over under again further once here there all any both each few
more most other some such no nor not only own same so than too very can will
just should now is am are was were be been being have has had having do does
did doing of as it its this that these those i you he she we they them his her
their our your my me him us""".split())

EXTENSIONS = {".txt", ".md", ".rst", ".py", ".js", ".ts", ".html", ".csv"}


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())
            if w not in STOPWORDS]


def load_documents(folder, recursive):
    docs = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in sorted(files):
                docs.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(folder)):
            docs.append(os.path.join(folder, name))
    loaded = []
    for path in docs:
        if not os.path.isfile(path):
            continue
        if os.path.splitext(path)[1].lower() not in EXTENSIONS:
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        if text.strip():
            loaded.append((path, text))
    return loaded


def build_index(documents):
    """Return (idf, doc_vectors, doc_norms) for TF-IDF cosine search."""
    doc_tokens = [tokenize(text) for _path, text in documents]
    doc_freq = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))
    total = len(documents)
    idf = {term: math.log((1 + total) / (1 + df)) + 1
           for term, df in doc_freq.items()}
    vectors, norms = [], []
    for tokens in doc_tokens:
        counts = Counter(tokens)
        vec = {term: (count / len(tokens)) * idf[term]
               for term, count in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        vectors.append(vec)
        norms.append(norm)
    return idf, vectors, norms


def search(query, documents, idf, vectors, norms):
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    counts = Counter(query_tokens)
    qvec = {term: (count / len(query_tokens)) * idf.get(term, 0.0)
            for term, count in counts.items() if term in idf}
    qnorm = math.sqrt(sum(v * v for v in qvec.values()))
    if qnorm == 0:
        return []
    scored = []
    for i, vec in enumerate(vectors):
        dot = sum(qvec[t] * vec.get(t, 0.0) for t in qvec)
        if norms[i] == 0:
            continue
        scored.append((dot / (qnorm * norms[i]), documents[i][0]))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def main():
    parser = argparse.ArgumentParser(
        description="Local TF-IDF search over a folder of documents "
                    "(no API needed).")
    parser.add_argument("folder", help="Folder of documents to search.")
    parser.add_argument("query", nargs="?", help="Search query. If omitted, "
                        "you will be prompted interactively.")
    parser.add_argument("-n", "--number", type=int, default=5,
                        help="Number of results to show (default: 5).")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Search subfolders too.")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Hide results below this similarity score.")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: {args.folder} is not a directory.", file=sys.stderr)
        return 1
    query = args.query
    if not query:
        query = input("Query: ").strip()
    if not query:
        print("Error: no query provided.", file=sys.stderr)
        return 1

    documents = load_documents(args.folder, args.recursive)
    if not documents:
        print(f"Error: no readable documents found in {args.folder}.",
              file=sys.stderr)
        return 1
    idf, vectors, norms = build_index(documents)
    results = search(query, documents, idf, vectors, norms)
    results = [(s, p) for s, p in results if s > args.min_score][:args.number]
    if not results:
        print("No matching documents found.")
        return 0
    for rank, (score, path) in enumerate(results, start=1):
        print(f"{rank}. {path}  (score {score:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
