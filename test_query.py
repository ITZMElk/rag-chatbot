"""Smoke-test script for retrieval and generation.

This script uses a hardcoded question, runs the retriever, passes the retrieved
chunks to the generator, and prints the final answer plus the sources used.

Usage:
    python test_query.py
"""

from generator import generate_answer
from retriever import retrieve_context


def main() -> None:
    # Replace this with a question that is relevant to your uploaded resume/PDF.
    question = "What experience does the candidate have in machine learning and software engineering?"

    print(f"Query: {question}\n")
    results = retrieve_context(question)

    if not results:
        print("No relevant chunks were retrieved.")
        return

    print(f"Retrieved {len(results)} chunk(s):")
    for index, chunk in enumerate(results, start=1):
        print(f"\n{index}. similarity={chunk['similarity']}")
        print(f"   source={chunk['source']}")
        print(f"   page={chunk['page']}")
        print(f"   chunk_index={chunk['chunk_index']}")
        print(f"   content={chunk['content'][:500]}{'...' if len(chunk['content']) > 500 else ''}")

    print("\nGenerating answer with Gemini...\n")
    response = generate_answer(question, results)
    print("Final answer:")
    print(response["answer"])

    print("\nSources used:")
    for index, source in enumerate(response.get("sources", []), start=1):
        print(
            f"{index}. source={source['source']} | page={source['page']} | similarity={source['similarity']}"
        )


if __name__ == "__main__":
    main()
