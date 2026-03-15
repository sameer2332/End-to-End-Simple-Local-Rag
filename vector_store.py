import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

model = SentenceTransformer("all-mpnet-base-v2")

client = chromadb.Client()

collection = client.get_or_create_collection(name="rag_collection")

df = pd.read_csv("text_chunks_and_embeddings_df.csv")

texts = df["sentence_chunk"].tolist()

embeddings = model.encode(texts).tolist()

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(texts))]
)


def retrieve_context(query):

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return " ".join(results["documents"][0])