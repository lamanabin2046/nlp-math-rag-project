"""
retrieve_demo.py
────────────────────────────────────────────────────────────────
Interactive demo that shows exactly which chunks your RAG system
retrieves for any query. Run this to satisfy comment #1:
"When a user runs a query we must be able to see the chunks
that our system retrieves."

Usage:
    python retrieve_demo.py
    python retrieve_demo.py --query "Find the area of a triangle with base 6cm and height 4cm"
    python retrieve_demo.py --query "Simplify algebraic fractions" --top_k 3

Output shows:
  - The query
  - Each retrieved chunk with chapter, page, type, similarity rank
  - The full RAG prompt that gets sent to the model
"""

import sys
import argparse
import numpy as np
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    CHROMA_DIR, CHROMA_COLLECTION,
    BERT_MODEL, CLIP_DIM, DESC_DIM,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5

RAG_PROMPT_TEMPLATE = """You are a mathematics tutor for Grade 10 students in Nepal.
Use the worked examples below as reference to solve the problem.
Show clear step-by-step working aligned with the Nepal curriculum.

{context}

Now solve this problem step by step:
Problem: {question}

Solution:"""


# ==============================================================================
# RETRIEVER
# ==============================================================================

def load_retriever():
    print("Loading embedding model ...")
    bert = SentenceTransformer(BERT_MODEL)

    print("Connecting to ChromaDB ...")
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(CHROMA_COLLECTION)
    print(f"Collection has {collection.count()} indexed chunks\n")

    return bert, collection


def retrieve_with_scores(question, bert, collection, top_k=DEFAULT_TOP_K):
    """
    Retrieve top-k chunks and return them with their similarity scores.
    """
    text_emb = bert.encode([question], normalize_embeddings=True)[0]
    pad      = np.zeros(CLIP_DIM + DESC_DIM, dtype=np.float32)
    query    = np.concatenate([text_emb, pad]).tolist()

    results = collection.query(
        query_embeddings=[query],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        # ChromaDB returns L2 distance — convert to cosine similarity
        similarity = 1 - (dist / 2)
        chunks.append({
            "rank":          i + 1,
            "similarity":    round(similarity, 4),
            "chapter_num":   meta.get("chapter_num",   "?"),
            "chapter_title": meta.get("chapter_title", "?"),
            "chapter_type":  meta.get("chapter_type",  "?"),
            "page":          meta.get("page",           "?"),
            "text":          doc,
        })

    return chunks


# ==============================================================================
# DISPLAY
# ==============================================================================

def display_chunks(query, chunks):
    """
    Pretty-print the retrieved chunks so they are easy to inspect.
    """
    print()
    print("=" * 70)
    print("QUERY")
    print("=" * 70)
    print(f"  {query}")

    print()
    print("=" * 70)
    print(f"TOP {len(chunks)} RETRIEVED CHUNKS")
    print("=" * 70)

    for chunk in chunks:
        print()
        print(f"  CHUNK {chunk['rank']}  (similarity score: {chunk['similarity']:.4f})")
        print(f"  {'-' * 60}")
        print(f"  Chapter : Unit {chunk['chapter_num']} - {chunk['chapter_title']}")
        print(f"  Type    : {chunk['chapter_type']}")
        print(f"  Page    : {chunk['page']}")
        print(f"  Content :")

        # Print the chunk text, indented, wrapped at 70 chars
        text = chunk["text"]
        lines = text.split("\n")
        for line in lines:
            # Word-wrap long lines
            while len(line) > 70:
                print(f"    {line[:70]}")
                line = line[70:]
            print(f"    {line}")

    print()
    print("=" * 70)


def display_rag_prompt(query, chunks):
    """
    Show the full prompt that would be sent to the model.
    """
    context_lines = ["Here are relevant worked examples from the Nepal Grade 10 "
                     "Mathematics textbook:\n"]
    for chunk in chunks:
        context_lines.append(
            f"Example {chunk['rank']} "
            f"(Unit {chunk['chapter_num']} - {chunk['chapter_title']}, "
            f"p.{chunk['page']}):"
        )
        context_lines.append(chunk["text"])
        context_lines.append("")

    context = "\n".join(context_lines)
    prompt  = RAG_PROMPT_TEMPLATE.format(context=context, question=query)

    print("=" * 70)
    print("FULL RAG PROMPT (sent to the model)")
    print("=" * 70)
    print(prompt)
    print("=" * 70)


# ==============================================================================
# INTERACTIVE MODE
# ==============================================================================

def interactive_mode(bert, collection, top_k):
    """
    Loop: user types a query, system shows retrieved chunks.
    Type 'quit' or press Ctrl+C to exit.
    """
    print("=" * 70)
    print("RAG RETRIEVAL DEMO  -  Interactive Mode")
    print("Type a math question to see which chunks are retrieved.")
    print("Commands:  'prompt' = show full RAG prompt  |  'quit' = exit")
    print("=" * 70)

    last_chunks = None
    last_query  = None

    while True:
        print()
        try:
            user_input = input("Enter query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Exiting.")
            break

        if user_input.lower() == "prompt":
            if last_chunks and last_query:
                display_rag_prompt(last_query, last_chunks)
            else:
                print("No query yet. Enter a question first.")
            continue

        # Retrieve and display
        print(f"\nRetrieving top {top_k} chunks ...")
        chunks = retrieve_with_scores(user_input, bert, collection, top_k)
        display_chunks(user_input, chunks)

        last_chunks = chunks
        last_query  = user_input

        # Quick summary table
        print("CHUNK SUMMARY")
        print(f"  {'Rank':<6} {'Similarity':<12} {'Chapter':<35} {'Page'}")
        print("  " + "-" * 60)
        for c in chunks:
            title = f"Unit {c['chapter_num']} - {c['chapter_title']}"
            print(f"  {c['rank']:<6} {c['similarity']:<12.4f} {title[:35]:<35} {c['page']}")
        print()
        print("Tip: type 'prompt' to see the full RAG prompt for this query.")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Show which chunks the RAG system retrieves for a query."
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=None,
        help="Query to retrieve chunks for. If not given, runs interactive mode."
    )
    parser.add_argument(
        "--top_k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})"
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Also print the full RAG prompt that would be sent to the model."
    )
    args = parser.parse_args()

    # Load retriever
    bert, collection = load_retriever()

    if args.query:
        # Single query mode
        print(f"Retrieving top {args.top_k} chunks ...")
        chunks = retrieve_with_scores(args.query, bert, collection, args.top_k)
        display_chunks(args.query, chunks)

        print("CHUNK SUMMARY")
        print(f"  {'Rank':<6} {'Similarity':<12} {'Chapter':<35} {'Page'}")
        print("  " + "-" * 60)
        for c in chunks:
            title = f"Unit {c['chapter_num']} - {c['chapter_title']}"
            print(f"  {c['rank']:<6} {c['similarity']:<12.4f} {title[:35]:<35} {c['page']}")

        if args.show_prompt:
            print()
            display_rag_prompt(args.query, chunks)

    else:
        # Interactive mode
        interactive_mode(bert, collection, args.top_k)


if __name__ == "__main__":
    main()