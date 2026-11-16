"""
Simple RAG demo (LLM + Retrieval) - runs in the terminal, no installs needed.

Pipeline: load docs -> chunk -> TF-IDF index -> retrieve top-k -> build prompt -> generate.
Generation: uses Claude if ANTHROPIC_API_KEY is set (pip install anthropic),
otherwise falls back to a built-in offline generator so you still get output.
"""
import math
import os
import re
from collections import Counter
from pathlib import Path

DOCS_DIR = Path(__file__).parent / "docs"
TOP_K = 3


STOP = {"what","is","are","the","a","an","of","in","to","for","does","do","why","how","and","or","used","can"}


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


def load_chunks():
    """Load every .txt file and split into sentence-based chunks."""
    chunks = []
    for f in sorted(DOCS_DIR.glob("*.txt")):
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                chunks.append({"source": f.name, "text": line})
    return chunks


class TfidfRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokens = [tokenize(c["text"]) for c in chunks]
        n = len(chunks)
        df = Counter(t for toks in self.tokens for t in set(toks))
        self.idf = {t: math.log((n + 1) / (d + 1)) + 1 for t, d in df.items()}
        self.vecs = [self._vec(toks) for toks in self.tokens]

    def _vec(self, toks):
        tf = Counter(toks)
        v = {t: c * self.idf.get(t, 0) for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1
        return {t: x / norm for t, x in v.items()}

    def search(self, query, k=TOP_K):
        q = self._vec(tokenize(query))
        scored = []
        for i, v in enumerate(self.vecs):
            score = sum(w * v.get(t, 0) for t, w in q.items())  # cosine similarity
            scored.append((score, i))
        scored.sort(reverse=True)
        return [(s, self.chunks[i]) for s, i in scored[:k] if s > 0]


def build_prompt(question, hits):
    context = "\n".join(f"[{i+1}] ({c['source']}) {c['text']}" for i, (_, c) in enumerate(hits))
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def generate(prompt, question, hits):
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text, "Claude API"
        except Exception as e:
            print(f"(LLM call failed: {e}; using offline generator)")
    # Offline fallback: pick the best-matching sentences from retrieved context.
    if not hits:
        return "I don't know - nothing relevant found in the documents.", "offline"
    qtok = set(tokenize(question))
    sentences = []
    for _, c in hits:
        sentences += re.split(r"(?<=[.!?])\s+", c["text"])
    sentences.sort(key=lambda s: len(qtok & set(tokenize(s))), reverse=True)
    return " ".join(sentences[:2]), "offline generator"


def main():
    chunks = load_chunks()
    retriever = TfidfRetriever(chunks)
    print(f"Indexed {len(chunks)} chunks from {DOCS_DIR.name}/\n")

    questions = [
        "What is RAG?",
        "Why does RAG reduce hallucinations?",
        "What are the steps in a RAG pipeline?",
        "What is cosine similarity used for?",
    ]
    for q in questions:
        hits = retriever.search(q)
        prompt = build_prompt(q, hits)
        answer, engine = generate(prompt, q, hits)
        print("=" * 70)
        print("Q:", q)
        print("Retrieved:")
        for score, c in hits:
            print(f"  - {score:.2f} [{c['source']}] {c['text'][:70]}...")
        print(f"A ({engine}): {answer}\n")


if __name__ == "__main__":
    main()
