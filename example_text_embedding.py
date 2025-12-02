"""
Example: Text Embedding to Doris with CocoIndex

This example demonstrates how to use the CocoIndex Doris connector
to build a document embedding pipeline that exports to Apache Doris.

Prerequisites:
1. Install CocoIndex: pip install cocoindex
2. Install Postgres for CocoIndex metadata: docker-compose up -d postgres
3. Set up Apache Doris cluster
4. Create the target table in Doris (see DDL below)

Table DDL for Doris:
```sql
CREATE DATABASE IF NOT EXISTS cocoindex_demo;

CREATE TABLE IF NOT EXISTS cocoindex_demo.document_embeddings (
    `filename` VARCHAR(1024),
    `location` VARCHAR(255),
    `text` TEXT,
    `embedding` ARRAY<FLOAT>
)
UNIQUE KEY(`filename`, `location`)
DISTRIBUTED BY HASH(`filename`, `location`) BUCKETS AUTO
PROPERTIES (
    "replication_num" = "1",
    "enable_unique_key_merge_on_write" = "true"
);

-- Optional: Create vector index for similarity search
ALTER TABLE cocoindex_demo.document_embeddings
ADD INDEX idx_embedding_vector (embedding) USING INVERTED;
```

Environment Variables:
- COCOINDEX_DATABASE_URL: PostgreSQL connection URL for CocoIndex metadata
- DORIS_FE_HOST: Doris FE host (default: localhost)
- DORIS_FE_PORT: Doris FE HTTP port (default: 8030)
- DORIS_USER: Doris username (default: root)
- DORIS_PASSWORD: Doris password (default: empty)
"""

import os
from datetime import timedelta
import cocoindex

# Import the Doris connector
from doris_target import DorisTarget


# =============================================================================
# Configuration
# =============================================================================

DORIS_CONFIG = {
    "fe_host": os.getenv("DORIS_FE_HOST", "localhost"),
    "fe_http_port": int(os.getenv("DORIS_FE_PORT", "8030")),
    "database": os.getenv("DORIS_DATABASE", "cocoindex_demo"),
    "table": os.getenv("DORIS_TABLE", "document_embeddings"),
    "username": os.getenv("DORIS_USER", "root"),
    "password": os.getenv("DORIS_PASSWORD", ""),
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MARKDOWN_FILES_PATH = "markdown_files"


# =============================================================================
# Flow Definition
# =============================================================================

@cocoindex.flow_def(name="DorisTextEmbedding")
def doris_text_embedding_flow(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope
) -> None:
    """
    Define an indexing flow that:
    1. Reads markdown files from a directory
    2. Splits them into chunks
    3. Generates embeddings for each chunk
    4. Exports to Doris
    """
    # Step 1: Add data source - read files from local directory
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path=MARKDOWN_FILES_PATH,
            included_patterns=["*.md", "*.txt"],
        ),
        refresh_interval=timedelta(seconds=10),  # Check for changes every 10s
    )
    
    # Step 2: Create a collector for the embeddings
    doc_embeddings = data_scope.add_collector()
    
    # Step 3: Process each document
    with data_scope["documents"].row() as doc:
        # Split document into chunks
        doc["chunks"] = doc["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown",
            chunk_size=1000,
            chunk_overlap=200,
        )
        
        # Step 4: Process each chunk
        with doc["chunks"].row() as chunk:
            # Generate embedding for the chunk
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SentenceTransformerEmbed(
                    model=EMBEDDING_MODEL
                )
            )
            
            # Collect the chunk data
            doc_embeddings.collect(
                filename=doc["filename"],
                location=chunk["location"],
                text=chunk["text"],
                embedding=chunk["embedding"],
            )
    
    # Step 5: Export to Doris
    doc_embeddings.export(
        "document_embeddings",
        DorisTarget(
            fe_host=DORIS_CONFIG["fe_host"],
            fe_http_port=DORIS_CONFIG["fe_http_port"],
            database=DORIS_CONFIG["database"],
            table=DORIS_CONFIG["table"],
            username=DORIS_CONFIG["username"],
            password=DORIS_CONFIG["password"],
            batch_size=5000,  # Batch size for Stream Load
        ),
        primary_key_fields=["filename", "location"],
    )


# =============================================================================
# Search Function
# =============================================================================

def search_similar_documents(
    query: str,
    top_k: int = 5,
    doris_config: dict = None
) -> list[dict]:
    """
    Search for similar documents using vector similarity in Doris.
    
    Note: This requires Doris with vector search capabilities (Doris 2.1+)
    or you can use inverted index for approximate search.
    
    Args:
        query: Search query text
        top_k: Number of results to return
        doris_config: Doris connection configuration
    
    Returns:
        List of matching documents with similarity scores
    """
    import requests
    from base64 import b64encode
    
    config = doris_config or DORIS_CONFIG
    
    # Generate query embedding using the same model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode(query).tolist()
    
    # Build Doris SQL query
    # Using L2 distance for similarity (lower is better)
    embedding_str = f"[{','.join(str(v) for v in query_embedding)}]"
    
    sql = f"""
    SELECT 
        filename,
        location,
        text,
        array_distance(embedding, {embedding_str}, 'cosine') as distance
    FROM {config['database']}.{config['table']}
    ORDER BY distance ASC
    LIMIT {top_k}
    """
    
    # Execute query via Doris HTTP API
    protocol = "http"
    base_url = f"{protocol}://{config['fe_host']}:{config['fe_http_port']}"
    url = f"{base_url}/api/query/default_cluster/{config['database']}"
    
    auth = b64encode(f"{config['username']}:{config['password']}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/json",
    }
    
    response = requests.post(
        url,
        json={"stmt": sql},
        headers=headers,
    )
    
    result = response.json()
    
    if result.get("status") != 0:
        raise RuntimeError(f"Query failed: {result.get('msg', 'Unknown error')}")
    
    # Parse results
    data = result.get("data", {})
    columns = [col["name"] for col in data.get("schema", [])]
    rows = data.get("data", [])
    
    return [
        {
            "filename": row[0],
            "location": row[1],
            "text": row[2],
            "score": 1 - float(row[3]),  # Convert distance to similarity
        }
        for row in rows
    ]


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point for the example."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CocoIndex Doris Text Embedding Example"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search query to find similar documents"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return for search"
    )
    
    args = parser.parse_args()
    
    if args.search:
        # Search mode
        print(f"Searching for: {args.search}")
        print("-" * 50)
        
        results = search_similar_documents(args.search, args.top_k)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['filename']} (score: {result['score']:.4f})")
            print(f"   Location: {result['location']}")
            print(f"   Text: {result['text'][:200]}...")
    else:
        # Index mode - handled by CocoIndex CLI
        print("Use CocoIndex CLI to run the indexing flow:")
        print("  cocoindex update --setup main")
        print("  cocoindex update -L main  # Live mode")
        print()
        print("Or use --search to query the index:")
        print("  python main.py --search 'your query here'")


if __name__ == "__main__":
    main()
