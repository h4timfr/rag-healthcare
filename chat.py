import sys

from src.exceptions import EmptyQueryError, InsufficientRetrievalError
from src.retrieval.hybrid_retriever import retrieve_for_question, retrieve_hybrid
from src.generation.prompt_builder import build_prompt
from src.generation.ollama_client import generate


def ask(query: str, history: list[dict]) -> str:
    # Retrieval never embeds or searches the raw conversation transcript -
    # that would blur the dense/keyword match against whatever earlier turns
    # happened to mention, not just the current question. retrieve_for_question
    # only uses `history` to resolve an elliptical reference ("what about the
    # first?", or a multi-part follow-up like "...second admission? What
    # about the first one?") into a standalone question BEFORE retrieval
    # runs, and to retrieve a genuinely multi-part question one part at a
    # time instead of as a single blended embedding - see its docstring and
    # intent.decompose_question(). A single-part question takes exactly the
    # old path: one retrieve_hybrid(query) call, unchanged.
    #
    # retrieve_hybrid raises plain ValueError (not a RagError) on empty query
    # and returns [] rather than raising when nothing is found; normalize
    # both to the same RagError-based contract the dense path already had.
    try:
        chunks = retrieve_for_question(query, history=history, retrieve_fn=retrieve_hybrid)
    except ValueError as e:
        raise EmptyQueryError(str(e)) from e

    if not chunks:
        raise InsufficientRetrievalError("No relevant chunks found for this query")

    # Conversation history is generation context, not a retrieval signal:
    # build_prompt() places it in a "RECENT CONVERSATION" section BEFORE
    # "USER QUESTION:"/"ANSWER:" so it can never bury the model's actual
    # generation cue behind more text (the previous version of this
    # function appended history after the prompt's trailing "ANSWER:").
    prompt = build_prompt(query, chunks, history=history, disable_benchmark_scope=True)

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