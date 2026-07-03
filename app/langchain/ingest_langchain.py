from .clients_langchain import vectorstore

def ingest_texts_langchain(texts: list[str]) -> None:
    vectorstore.add_texts(texts)

def remove_docs_langchain() -> None: # python -c "from ingest import remove_docs; remove_docs()"
    vectorstore.client.delete_collection("rag_context_langchain")

if __name__ == "__main__":
    texts = [
        # Basic facts
        "The quick brown fox jumps over the lazy dog.",
        "Paris is the capital of France and has a population of 2.1 million.",
        "Mount Everest is 8,848 meters tall and was first climbed in 1953 by Edmund Hillary and Tenzing Norgay.",

        # Multi-fact sentences
        "Elon Musk founded SpaceX in 2002 and Tesla in 2003, and his net worth is approximately $200 billion.",
        "The Eiffel Tower was completed in 1889, weighs 10,100 tonnes, and attracts 7 million visitors annually.",
        "Python was created by Guido van Rossum and first released in 1991.",

        # Technical data
        "The speed of light in vacuum is 299,792,458 meters per second.",
        "Water boils at 100°C at standard atmospheric pressure.",
        "The human body has 206 bones.",

        # Dates & events
        "World War II ended on September 2, 1945.",
        "The first iPhone was released on June 29, 2007.",
        "The Berlin Wall fell on November 9, 1989.",

        # Fiction/creative
        "In the novel 1984, the main character Winston Smith works at the Ministry of Truth.",
        "The Three-Body Problem is a science fiction novel written by Liu Cixin.",
        "Sherlock Holmes lives at 221B Baker Street in London."
    ]
    print("Running example ingestion...")
    ingest_texts_langchain(texts)
