"""
Complete RAG Pipeline Example with CocoIndex and Doris

This example demonstrates a full RAG (Retrieval-Augmented Generation) pipeline:
1. Index documents from various sources
2. Export embeddings to Apache Doris
3. Perform semantic search
4. Generate responses using LLM

The pipeline automatically handles incremental updates - when documents
change, only the affected chunks are re-embedded and updated in Doris.
"""

import os
import dataclasses
from datetime import timedelta
from typing import Optional

import cocoindex
from doris_target import DorisTarget


# =============================================================================
# Configuration
# =============================================================================

@dataclasses.dataclass
class Config:
    """Pipeline configuration."""
    # Doris settings
    doris_host: str = os.getenv("DORIS_FE_HOST", "localhost")
    doris_port: int = int(os.getenv("DORIS_FE_PORT", "8030"))
    doris_database: str = os.getenv("DORIS_DATABASE", "rag_demo")
    doris_user: str = os.getenv("DORIS_USER", "root")
    doris_password: str = os.getenv("DORIS_PASSWORD", "")
    
    # Document settings
    docs_path: str = os.getenv("DOCS_PATH", "./documents")
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 200
    
    # Search settings
    search_top_k: int = 5


CONFIG = Config()


# =============================================================================
# Doris Table Schema
# =============================================================================

DORIS_TABLE_DDL = """
-- Create database
CREATE DATABASE IF NOT EXISTS {database};

-- Create main embeddings table
CREATE TABLE IF NOT EXISTS {database}.document_chunks (
    `doc_id` VARCHAR(512) COMMENT 'Document identifier',
    `chunk_id` VARCHAR(255) COMMENT 'Chunk position/location',
    `source_path` VARCHAR(1024) COMMENT 'Original file path',
    `title` VARCHAR(500) COMMENT 'Document title',
    `text` TEXT COMMENT 'Chunk text content',
    `embedding` ARRAY<FLOAT> COMMENT 'Text embedding vector',
    `metadata` JSON COMMENT 'Additional metadata',
    `indexed_at` DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT 'Index timestamp'
)
UNIQUE KEY(`doc_id`, `chunk_id`)
DISTRIBUTED BY HASH(`doc_id`) BUCKETS AUTO
PROPERTIES (
    "replication_num" = "1",
    "enable_unique_key_merge_on_write" = "true",
    "store_row_column" = "true"
);

-- Create inverted index for full-text search
CREATE INDEX IF NOT EXISTS idx_text_fulltext 
ON {database}.document_chunks (text) USING INVERTED 
PROPERTIES("parser" = "unicode", "support_phrase" = "true");

-- Create inverted index for vector search (if supported)
-- ALTER TABLE {database}.document_chunks 
-- ADD INDEX idx_embedding_vector (embedding) USING INVERTED;
"""


# =============================================================================
# Custom Functions
# =============================================================================

@cocoindex.op.function()
def extract_title(filename: str, content: str) -> str:
    """Extract document title from filename or content."""
    # Try to extract from markdown header
    lines = content.split('\n')
    for line in lines[:10]:
        if line.startswith('# '):
            return line[2:].strip()
    
    # Fall back to filename
    name = os.path.splitext(os.path.basename(filename))[0]
    return name.replace('_', ' ').replace('-', ' ').title()


@cocoindex.op.function()
def create_doc_id(filename: str) -> str:
    """Create a stable document ID from filename."""
    import hashlib
    # Use hash for consistent, URL-safe IDs
    return hashlib.md5(filename.encode()).hexdigest()[:16]


@cocoindex.op.function()
def extract_metadata(filename: str, content: str) -> dict:
    """Extract metadata from document."""
    import re
    
    metadata = {
        "file_extension": os.path.splitext(filename)[1],
        "char_count": len(content),
        "word_count": len(content.split()),
    }
    
    # Try to extract YAML frontmatter
    frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
    if frontmatter_match:
        try:
            import yaml
            fm = yaml.safe_load(frontmatter_match.group(1))
            if isinstance(fm, dict):
                metadata.update(fm)
        except Exception:
            pass
    
    return metadata


# =============================================================================
# Flow Definition
# =============================================================================

@cocoindex.flow_def(name="DorisRAG")
def doris_rag_flow(
    flow_builder: cocoindex.FlowBuilder,
    data_scope: cocoindex.DataScope
) -> None:
    """
    Define the RAG indexing flow.
    
    This flow:
    1. Reads documents from local directory
    2. Extracts metadata and generates embeddings
    3. Exports to Doris for vector search
    """
    # Step 1: Add data source
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(
            path=CONFIG.docs_path,
            included_patterns=["*.md", "*.txt", "*.rst"],
        ),
        refresh_interval=timedelta(seconds=30),
    )
    
    # Step 2: Create collector
    chunk_collector = data_scope.add_collector()
    
    # Step 3: Process each document
    with data_scope["documents"].row() as doc:
        # Extract document-level information
        doc["doc_id"] = doc["filename"].transform(create_doc_id)
        doc["title"] = flow_builder.transform(
            extract_title,
            filename=doc["filename"],
            content=doc["content"]
        )
        doc["metadata"] = flow_builder.transform(
            extract_metadata,
            filename=doc["filename"],
            content=doc["content"]
        )
        
        # Split into chunks
        doc["chunks"] = doc["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown",
            chunk_size=CONFIG.chunk_size,
            chunk_overlap=CONFIG.chunk_overlap,
        )
        
        # Process each chunk
        with doc["chunks"].row() as chunk:
            # Generate embedding
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SentenceTransformerEmbed(
                    model=CONFIG.embedding_model
                )
            )
            
            # Collect chunk data
            chunk_collector.collect(
                doc_id=doc["doc_id"],
                chunk_id=chunk["location"],
                source_path=doc["filename"],
                title=doc["title"],
                text=chunk["text"],
                embedding=chunk["embedding"],
                metadata=doc["metadata"],
            )
    
    # Step 4: Export to Doris
    chunk_collector.export(
        "document_chunks",
        DorisTarget(
            fe_host=CONFIG.doris_host,
            fe_http_port=CONFIG.doris_port,
            database=CONFIG.doris_database,
            table="document_chunks",
            username=CONFIG.doris_user,
            password=CONFIG.doris_password,
            batch_size=2000,
        ),
        primary_key_fields=["doc_id", "chunk_id"],
    )


# =============================================================================
# Search and RAG Functions
# =============================================================================

class DorisSearchClient:
    """Client for searching documents in Doris."""
    
    def __init__(self, config: Config = CONFIG):
        self.config = config
        self._model = None
    
    @property
    def model(self):
        """Lazy load embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.config.embedding_model)
        return self._model
    
    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query."""
        return self.model.encode(query).tolist()
    
    def _execute_query(self, sql: str) -> list[dict]:
        """Execute a SQL query against Doris."""
        import requests
        from base64 import b64encode
        
        url = f"http://{self.config.doris_host}:{self.config.doris_port}/api/query/default_cluster/{self.config.doris_database}"
        
        auth = b64encode(
            f"{self.config.doris_user}:{self.config.doris_password}".encode()
        ).decode()
        
        response = requests.post(
            url,
            json={"stmt": sql},
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        
        result = response.json()
        
        if result.get("status") != 0:
            raise RuntimeError(f"Query failed: {result.get('msg', 'Unknown error')}")
        
        data = result.get("data", {})
        columns = [col["name"] for col in data.get("schema", [])]
        rows = data.get("data", [])
        
        return [dict(zip(columns, row)) for row in rows]
    
    def semantic_search(
        self,
        query: str,
        top_k: int = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            filters: Additional filter conditions
        
        Returns:
            List of matching documents with scores
        """
        top_k = top_k or self.config.search_top_k
        
        # Generate query embedding
        query_embedding = self.embed_query(query)
        embedding_str = f"[{','.join(str(v) for v in query_embedding)}]"
        
        # Build SQL with vector distance
        where_clause = ""
        if filters:
            conditions = [f"{k} = '{v}'" for k, v in filters.items()]
            where_clause = "WHERE " + " AND ".join(conditions)
        
        sql = f"""
        SELECT 
            doc_id,
            chunk_id,
            source_path,
            title,
            text,
            metadata,
            array_distance(embedding, {embedding_str}, 'cosine') as distance
        FROM {self.config.doris_database}.document_chunks
        {where_clause}
        ORDER BY distance ASC
        LIMIT {top_k}
        """
        
        results = self._execute_query(sql)
        
        # Convert distance to similarity score
        for r in results:
            r["score"] = 1 - float(r.get("distance", 1))
        
        return results
    
    def fulltext_search(
        self,
        query: str,
        top_k: int = None,
    ) -> list[dict]:
        """
        Perform full-text search using Doris inverted index.
        
        Args:
            query: Search query text
            top_k: Number of results to return
        
        Returns:
            List of matching documents
        """
        top_k = top_k or self.config.search_top_k
        
        # Escape query for MATCH_ANY
        escaped_query = query.replace("'", "''")
        
        sql = f"""
        SELECT 
            doc_id,
            chunk_id,
            source_path,
            title,
            text,
            metadata
        FROM {self.config.doris_database}.document_chunks
        WHERE text MATCH_ANY '{escaped_query}'
        LIMIT {top_k}
        """
        
        return self._execute_query(sql)
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        semantic_weight: float = 0.7,
    ) -> list[dict]:
        """
        Perform hybrid search combining semantic and full-text search.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
        
        Returns:
            List of matching documents with combined scores
        """
        top_k = top_k or self.config.search_top_k
        fulltext_weight = 1 - semantic_weight
        
        # Get both result sets
        semantic_results = self.semantic_search(query, top_k * 2)
        fulltext_results = self.fulltext_search(query, top_k * 2)
        
        # Combine results using reciprocal rank fusion
        combined = {}
        
        for i, result in enumerate(semantic_results):
            key = (result["doc_id"], result["chunk_id"])
            rrf_score = semantic_weight / (60 + i)
            combined[key] = {
                **result,
                "combined_score": rrf_score,
            }
        
        for i, result in enumerate(fulltext_results):
            key = (result["doc_id"], result["chunk_id"])
            rrf_score = fulltext_weight / (60 + i)
            if key in combined:
                combined[key]["combined_score"] += rrf_score
            else:
                combined[key] = {
                    **result,
                    "score": 0,  # No semantic score
                    "combined_score": rrf_score,
                }
        
        # Sort by combined score and return top_k
        results = sorted(
            combined.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )[:top_k]
        
        return results


def generate_rag_response(
    query: str,
    search_client: DorisSearchClient = None,
    llm_model: str = "gpt-3.5-turbo",
) -> dict:
    """
    Generate a RAG response using retrieved context.
    
    Args:
        query: User query
        search_client: Search client instance
        llm_model: LLM model to use for generation
    
    Returns:
        Dictionary with response and source documents
    """
    client = search_client or DorisSearchClient()
    
    # Retrieve relevant documents
    results = client.hybrid_search(query)
    
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(results, 1):
        context_parts.append(
            f"[{i}] {doc['title']} ({doc['source_path']})\n{doc['text']}"
        )
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Generate response using LLM
    prompt = f"""Based on the following context, answer the user's question.
If the answer is not in the context, say so.

Context:
{context}

Question: {query}

Answer:"""
    
    # Here you would call your LLM API
    # This is a placeholder - implement with your preferred LLM
    response_text = f"[LLM Response placeholder for query: {query}]"
    
    return {
        "query": query,
        "response": response_text,
        "sources": results,
        "context_used": len(results),
    }


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main CLI interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CocoIndex Doris RAG Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Schema command
    schema_parser = subparsers.add_parser("schema", help="Print Doris table DDL")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--mode",
        choices=["semantic", "fulltext", "hybrid"],
        default="hybrid",
        help="Search mode"
    )
    search_parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    # RAG command
    rag_parser = subparsers.add_parser("rag", help="Generate RAG response")
    rag_parser.add_argument("query", help="Question to answer")
    
    args = parser.parse_args()
    
    if args.command == "schema":
        print(DORIS_TABLE_DDL.format(database=CONFIG.doris_database))
    
    elif args.command == "search":
        client = DorisSearchClient()
        
        if args.mode == "semantic":
            results = client.semantic_search(args.query, args.top_k)
        elif args.mode == "fulltext":
            results = client.fulltext_search(args.query, args.top_k)
        else:
            results = client.hybrid_search(args.query, args.top_k)
        
        print(f"\nSearch results for: {args.query}")
        print("=" * 60)
        
        for i, doc in enumerate(results, 1):
            score = doc.get("combined_score") or doc.get("score", 0)
            print(f"\n{i}. [{score:.4f}] {doc['title']}")
            print(f"   Source: {doc['source_path']}")
            print(f"   Text: {doc['text'][:200]}...")
    
    elif args.command == "rag":
        result = generate_rag_response(args.query)
        
        print(f"\nQuestion: {result['query']}")
        print("=" * 60)
        print(f"\nAnswer:\n{result['response']}")
        print(f"\n\nSources ({result['context_used']} documents used):")
        for i, src in enumerate(result["sources"], 1):
            print(f"  {i}. {src['title']} ({src['source_path']})")
    
    else:
        parser.print_help()
        print("\n\nTo index documents, use CocoIndex CLI:")
        print("  cocoindex update --setup rag_example")
        print("  cocoindex update -L rag_example  # Live mode")


if __name__ == "__main__":
    main()
