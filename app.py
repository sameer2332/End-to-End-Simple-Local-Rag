from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
from cache import get_cache, set_cache
from vector_store import retrieve_context

app = FastAPI(title="RAG API")

embedding_model = None


@app.on_event("startup")
def load_model():
    global embedding_model
    embedding_model = SentenceTransformer("all-mpnet-base-v2")


@app.get("/")
def health():
    return {"status": "RAG API running"}


@app.post("/ask")
def ask(query: str):

    # Check Redis cache
    cached = get_cache(query)
    if cached:
        return {
            "query": query,
            "context": cached,
            "source": "redis-cache"
        }

    # Retrieve from vector DB
    context = retrieve_context(query)

    # Save to cache
    set_cache(query, context)

    return {
        "query": query,
        "context": context,
        "source": "vector-db"
    }