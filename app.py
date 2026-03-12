from fastapi import FastAPI
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
from cache import get_cache, set_cache

app = FastAPI(title="RAG API")
device = "cpu"

embedding_model = SentenceTransformer(
    model_name_or_path="all-mpnet-base-v2", device=device
)

df = pd.read_csv("text_chunks_and_embeddings_df.csv")

embeddings = torch.tensor(
    df["embedding"]
    .apply(lambda x: [float(i.strip().replace(",", "")) for i in x.strip("[]").split()])
    .to_list()
)


text_chunks = df["sentence_chunk"].tolist()


def retrieve_context(query: str, top_k: int = 3):
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)

    scores = util.dot_score(query_embedding, embeddings)[0]
    top_results = torch.topk(scores, k=top_k)

    context = []
    for idx in top_results.indices:
        context.append(text_chunks[int(idx)])

    return " ".join(context)


@app.get("/")
def health():
    return {"status": "RAG API running"}


@app.post("/ask")
def ask(query: str):

    #  Check Redis cache
    cached = get_cache(query)
    if cached:
        return {"query": query, "context": cached, "source": "redis-cache"}

    # If not cached → compute
    context = retrieve_context(query)

    # Save in Redis
    set_cache(query, context)

    return {"query": query, "context": context, "source": "computed"}
