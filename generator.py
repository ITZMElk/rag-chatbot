"""Answer generation layer for the RAG chatbot.

The generator does not rely on the model's general knowledge alone. Instead,
it is given only the retrieved chunks from the vector database, which makes the
response more grounded and easier to explain to a learner.
"""

from __future__ import annotations

import os
from typing import Sequence

from config import GEMINI_MODEL_NAME


def build_prompt(query: str, context_chunks: Sequence[dict[str, object]]) -> str:
    """Create a grounded prompt that includes the retrieved chunks as context."""
    context_block = []
    for index, chunk in enumerate(context_chunks, start=1):
        context_block.append(
            f"Source {index}: {chunk['source']} (page {chunk['page']})\n{chunk['content']}"
        )

    context_text = "\n\n".join(context_block)
    return (
        "You are a grounded assistant. Answer only using the provided context. "
        "If the answer is not present in the context, say that you do not have enough "
        "information and do not make anything up.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def generate_answer(query: str, context_chunks: Sequence[dict[str, object]]) -> dict[str, object]:
    """Call Gemini with the retrieved context and return the answer plus sources."""
    if not context_chunks:
        return {
            "answer": "I do not have enough relevant context to answer that question confidently.",
            "sources": [],
            "grounded": False,
        }

    try:
        import google.generativeai as genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set in the .env file.")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=GEMINI_MODEL_NAME)
        prompt = build_prompt(query, context_chunks)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 700},
        )
        answer_text = getattr(response, "text", None) or "I could not generate an answer."
        return {
            "answer": answer_text,
            "sources": list(context_chunks),
            "grounded": True,
        }
    except Exception as exc:  # pragma: no cover - defensive UI handling
        raise RuntimeError(f"Gemini request failed: {exc}") from exc
