import os
import functools
from dotenv import load_dotenv
import cocoindex
from doris_vector_search import DorisVectorClient, AuthOptions, IndexOptions
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
# Using OpenRouter Embeddings API
from cocoindex.llm import LlmApiType
from cocoindex.auth_registry import add_transient_auth_entry

# Use the Doris target defined in this repo
from doris_target import DorisTarget

# Load environment variables
load_dotenv()

# Doris connection configuration (override via environment variables)
DORIS_FE_HOST = os.getenv("DORIS_FE_HOST", "localhost")
DORIS_FE_PORT = int(os.getenv("DORIS_FE_PORT", "5937"))
DORIS_QUERY_PORT = int(os.getenv("DORIS_QUERY_PORT", "6937"))
DORIS_DATABASE = os.getenv("DORIS_DATABASE", "cocoindex_demo")
DORIS_TABLE = os.getenv("DORIS_TABLE", "document_embeddings")
DORIS_USER = os.getenv("DORIS_USER", "root")
DORIS_PASSWORD = os.getenv("DORIS_PASSWORD", "")

# Source directory for demo documents
DOCS_DIR = os.getenv("DOCS_DIR", "/mnt/disk4/hezhiqiang/code/cocoindex/examples/text_embedding_qdrant/markdown_files")

# OpenRouter embedding config (must match at index and query time)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/embeddings")
OPENROUTER_API_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")

@cocoindex.transform_flow()
def text_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[list[float]]:
    """
    Shared embedding step using OpenAI Embedding API.
    """
    # Use CocoIndex built-in EmbedText function with OpenRouter
    return text.transform(
        cocoindex.functions.EmbedText(
            api_type=LlmApiType.OPEN_ROUTER,
            model=OPENROUTER_EMBED_MODEL,
            address=OPENROUTER_API_BASE,
            output_dimension=4096,
            api_key=add_transient_auth_entry(OPENROUTER_API_KEY),
        )
    )


@cocoindex.flow_def(name="TextEmbeddingWithDoris")
def text_embedding_flow(
    flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope
) -> None:
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path=DOCS_DIR,
            included_patterns=["**/*.md", "**/*.txt"],
            excluded_patterns=["**/*.pdf", "**/*.png", "**/*.jpg", "**/*.jpeg"],
        )
    )

    doc_embeddings = data_scope.add_collector()

    with data_scope["documents"].row() as doc:
        doc["chunks"] = doc["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown",
            chunk_size=2000,
            chunk_overlap=500,
        )

        with doc["chunks"].row() as chunk:
            chunk["embedding"] = text_to_embedding(chunk["text"])
            doc_embeddings.collect(
                id=cocoindex.GeneratedField.UUID,
                filename=doc["filename"],
                location=chunk["location"],
                text=chunk["text"],
                embedding=chunk["embedding"],
            )

    doc_embeddings.export(
        "doc_embeddings",
        DorisTarget(
            fe_host=DORIS_FE_HOST,
            fe_http_port=DORIS_FE_PORT,
            query_port=DORIS_QUERY_PORT,
            database=DORIS_DATABASE,
            table=DORIS_TABLE,
            username=DORIS_USER,
            password=DORIS_PASSWORD,
            batch_size=5000,
        ),
        primary_key_fields=["id"],
    )


@functools.cache
def _open_doris_table():
    """
    Open Doris table via doris-vector-search SDK.
    Requires the table to exist and contain a vector column (e.g., `embedding`).
    """
    client = DorisVectorClient(
        DORIS_DATABASE,
        auth_options=AuthOptions(
            host=DORIS_FE_HOST,
            http_port=DORIS_FE_PORT,
            query_port=DORIS_QUERY_PORT,
            user=DORIS_USER,
            password=DORIS_PASSWORD,
        ),
    )
    return client.open_table(DORIS_TABLE)

def _openrouter_embed(text: str) -> list[float]:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("Missing OPENROUTER_API_KEY")
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENROUTER_EMBED_MODEL,
        "input": text,
    }
    resp = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    return j["data"][0]["embedding"]


@text_embedding_flow.query_handler(
    result_fields=cocoindex.QueryHandlerResultFields(
        embedding=["embedding"],
        score="score",
    ),
)

def search(query: str) -> cocoindex.QueryOutput:
    # Compute embedding for the input query using OpenRouter Embeddings
    query_embedding = _openrouter_embed(query)

    # Use doris-vector-search SDK to perform vector search
    table = _open_doris_table()
    df = (
        table
        .search(query_embedding, vector_column="embedding")
        .limit(10)
        .to_pandas()
    )

    # doris-vector-search returns a DataFrame with columns from the table
    # and a `distance` column. Convert distance to a similarity score.
    records = df.to_dict("records") if hasattr(df, "to_dict") else []

    def _to_score(distance: float) -> float:
        # Generic mapping for distance to similarity
        try:
            d = float(distance)
            return 1.0 / (1.0 + d)
        except Exception:
            return 0.0

    results = []
    for row in records:
        results.append({
            "id": row.get("id"),
            "filename": row.get("filename"),
            "location": row.get("location"),
            "text": row.get("text"),
            "embedding": row.get("embedding"),
            "score": _to_score(row.get("distance", 0.0)),
        })

    return cocoindex.QueryOutput(
        results=results,
        query_info=cocoindex.QueryInfo(
            embedding=query_embedding,
            similarity_metric=cocoindex.VectorSimilarityMetric.INNER_PRODUCT,
        ),
    )


def _main() -> None:
    # Interactive search loop
    while True:
        query = input("Enter search query (or Enter to quit): ")
        if query == "":
            break

        out = search(query)
        print("\nSearch results:")
        for r in out.results:
            print(f"[{r['score']:.3f}] {r.get('filename', r.get('id'))}")
            print(f"    {r['text']}")
            print("---")
        print()


if __name__ == "__main__":
    _main()
