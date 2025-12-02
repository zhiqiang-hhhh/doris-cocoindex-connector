# CocoIndex Doris Connector

Apache Doris connector for [CocoIndex](https://github.com/cocoindex-io/cocoindex) - enables exporting embeddings, documents, and structured data to Apache Doris for vector search and analytics.

## Features

- **Stream Load Integration**: High-performance batch data ingestion via Doris Stream Load API
- **UNIQUE KEY Model Support**: Automatic upsert and delete operations
- **Vector Support**: Export embeddings as `ARRAY<FLOAT>` for vector similarity search
- **Incremental Updates**: Seamlessly integrates with CocoIndex's incremental processing
- **Batch Processing**: Configurable batch sizes for optimal performance

## Installation

```bash
pip install cocoindex-doris-connector

# Or install from source
pip install -e .
```

## Prerequisites

1. **CocoIndex**: Install and configure CocoIndex
   ```bash
   pip install cocoindex
   ```

2. **PostgreSQL**: For CocoIndex metadata storage
   ```bash
   docker run -d --name postgres -p 5432:5432 \
     -e POSTGRES_PASSWORD=cocoindex \
     postgres:15
   ```

3. **Apache Doris**: A running Doris cluster (version 2.0+ recommended for vector search)

## Quick Start

### 1. Create the Doris Table

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
```

### 2. Define Your Flow

```python
import cocoindex
from doris_target import DorisTarget

@cocoindex.flow_def(name="MyDorisFlow")
def my_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    # Add data source
    data_scope["documents"] = flow_builder.add_source(
        cocoindex.sources.LocalFile(path="./docs")
    )
    
    # Create collector
    doc_embeddings = data_scope.add_collector()
    
    # Process documents
    with data_scope["documents"].row() as doc:
        doc["chunks"] = doc["content"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown",
            chunk_size=1000,
        )
        
        with doc["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].transform(
                cocoindex.functions.SentenceTransformerEmbed(
                    model="sentence-transformers/all-MiniLM-L6-v2"
                )
            )
            
            doc_embeddings.collect(
                filename=doc["filename"],
                location=chunk["location"],
                text=chunk["text"],
                embedding=chunk["embedding"],
            )
    
    # Export to Doris
    doc_embeddings.export(
        "document_embeddings",
        DorisTarget(
            fe_host="localhost",
            fe_http_port=8030,
            database="cocoindex_demo",
            table="document_embeddings",
            username="root",
            password="",
        ),
        primary_key_fields=["filename", "location"],
    )
```

### 3. Run the Flow

```bash
# Initial setup and index
cocoindex update --setup main

# Live mode (continuous updates)
cocoindex update -L main
```

### 4. Query in Doris

```sql
-- Full-text search
SELECT filename, text 
FROM cocoindex_demo.document_embeddings 
WHERE text MATCH_ANY 'your search terms';

-- Vector similarity search (Doris 2.1+)
SELECT 
    filename,
    text,
    array_distance(embedding, [0.1, 0.2, ...], 'cosine') as distance
FROM cocoindex_demo.document_embeddings
ORDER BY distance ASC
LIMIT 10;
```

## Configuration Options

### DorisTarget Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fe_host` | str | required | Doris FE host address |
| `fe_http_port` | int | 8030 | Doris FE HTTP port |
| `database` | str | required | Target database name |
| `table` | str | required | Target table name |
| `username` | str | "root" | Doris username |
| `password` | str | "" | Doris password |
| `enable_https` | bool | False | Use HTTPS for connections |
| `stream_load_timeout` | int | 600 | Stream Load timeout (seconds) |
| `batch_size` | int | 10000 | Rows per Stream Load batch |
| `auto_create_table` | bool | True | Auto-create table if not exists |
| `replication_num` | int | 1 | Replication for auto-created tables |

### Environment Variables

```bash
export COCOINDEX_DATABASE_URL="postgresql://user:pass@localhost:5432/cocoindex"
export DORIS_FE_HOST="localhost"
export DORIS_FE_PORT="8030"
export DORIS_USER="root"
export DORIS_PASSWORD=""
export DORIS_DATABASE="cocoindex_demo"
```

## Advanced Usage

### Custom Data Types

```python
from doris_target import DorisTarget, create_doris_table_ddl

# Generate DDL for custom schema
ddl = create_doris_table_ddl(
    database="my_db",
    table="my_table",
    schema={
        "id": "BIGINT",
        "title": "VARCHAR(500)",
        "content": "TEXT",
        "embedding": "ARRAY<FLOAT>",
        "metadata": "JSON",
        "created_at": "DATETIME",
    },
    primary_keys=["id"],
    vector_fields={"embedding": 768},
    replication_num=3,
)
print(ddl)
```

### Multiple Targets

```python
# Export to multiple Doris tables
embeddings.export("table1", DorisTarget(..., table="embeddings"))
metadata.export("table2", DorisTarget(..., table="metadata"))
```

### With Vector Index

For vector similarity search, create an inverted index on the embedding column:

```sql
ALTER TABLE cocoindex_demo.document_embeddings
ADD INDEX idx_embedding_vector (embedding) USING INVERTED;
```

## Best Practices

### Table Design

1. **Use UNIQUE KEY model** for upsert/delete support
2. **Enable merge-on-write** for better write performance:
   ```sql
   "enable_unique_key_merge_on_write" = "true"
   ```
3. **Choose appropriate bucket strategy** based on data distribution

### Performance Tuning

1. **Batch Size**: Adjust `batch_size` based on your data size and network
2. **Timeout**: Increase `stream_load_timeout` for large batches
3. **Parallel Loading**: CocoIndex handles parallelism automatically

### Schema Evolution

When modifying schema:
1. Add new columns via `ALTER TABLE`
2. Update your CocoIndex flow to collect new fields
3. Re-run the flow with `cocoindex update --setup`

## Troubleshooting

### Common Issues

**Stream Load Timeout**
```python
DorisTarget(
    ...,
    stream_load_timeout=1200,  # Increase timeout
    batch_size=5000,           # Reduce batch size
)
```

**Connection Refused**
- Check Doris FE is running and HTTP port is accessible
- Verify firewall settings

**Authentication Failed**
- Double-check username/password
- Ensure user has INSERT/DELETE privileges on the target table

### Debug Mode

Enable logging for detailed Stream Load responses:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

Apache License 2.0

## See Also

- [CocoIndex Documentation](https://cocoindex.io/docs)
- [Apache Doris Documentation](https://doris.apache.org/docs)
- [Stream Load Reference](https://doris.apache.org/docs/data-operate/import/stream-load-manual)
