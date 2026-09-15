import sys

from src.exceptions import EmptyQueryError, InsufficientRetrievalError
from src.retrieval.hybrid_retriever import retrieve_hybrid
from src.generation.prompt_builder import build_prompt
from src.generation.ollama_client import generate


def ask(query: str, history: list[dict]) -> str:
    # Make follow-up questions understandable to retrieval.
    retrieval_query = query

    if history:
        previous = history[-4:]
        previous_text = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in previous
        )
        retrieval_query = (
            f"Conversation context:\n{previous_text}\n\n"
            f"Current question:\n{query}"
        )

    # retrieve_hybrid raises plain ValueError (not a RagError) on empty query
    # and returns [] rather than raising when nothing is found; normalize both
    # to the same RagError-based contract the dense path already had.
    try:
        chunks = retrieve_hybrid(retrieval_query)
    except ValueError as e:
        raise EmptyQueryError(str(e)) from e

    if not chunks:
        raise InsufficientRetrievalError("No relevant chunks found for this query")

    prompt = build_prompt(query, chunks)

    if history:
        history_text = "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in history[-4:]
        )
        prompt = (
            f"{prompt}\n\n"
            f"RECENT CONVERSATION:\n{history_text}\n"
        )

    answer = generate(prompt)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")
    seen = set()
    source_number = 1

    for chunk in chunks:
        key = (chunk["source"], chunk["page"])

        if key not in seen:
            print(f"[{source_number}] {chunk['source']} — page {chunk['page']}")
            seen.add(key)
            source_number += 1

    return answer


def main() -> None:
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:]).strip()
        ask(query, [])
        return

    print("Healthcare RAG Chat")
    print("Type 'exit', 'quit', or 'clear' to manage the session.\n")

    history: list[dict] = []

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        if query.lower() == "clear":
            history.clear()
            print("Conversation cleared.\n")
            continue

        try:
            answer = ask(query, history)

            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()