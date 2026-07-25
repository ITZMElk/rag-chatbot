from retriever import get_collection, retrieve_context

print("starting retrieval check")
collection = get_collection()
print("COLLECTION_COUNT:", collection.count())
print("COLLECTION_GET:", collection.get(include=["documents", "metadatas"]))

queries = ["what is resume analyzer", "disadvantages in the resume"]
for question in queries:
    print(f"QUERY_START: {question}")
    try:
        results = retrieve_context(question)
        print(f"COUNT: {len(results)}")
        for item in results:
            print({
                "similarity": item["similarity"],
                "source": item["source"],
                "page": item["page"],
                "chunk_index": item["chunk_index"],
            })
    except Exception as exc:
        print("ERROR:", repr(exc))
    print("---")
