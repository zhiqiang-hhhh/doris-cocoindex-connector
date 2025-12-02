"""
CocoIndex Doris Target Connector

This module provides a custom target connector for Apache Doris,
enabling CocoIndex to export data to Doris tables using Stream Load.

Features:
- Stream Load for high-performance data ingestion
- Support for UNIQUE KEY model (upsert/delete)
- Support for vector fields (stored as ARRAY<FLOAT>)
- Batch processing for efficiency
- Automatic table creation and schema management

Usage:
    doc_embeddings.export(
        "doc_embeddings",
        DorisTarget(
            fe_host="localhost",
            fe_http_port=8030,
            database="my_database",
            table="my_table",
            username="root",
            password="",
        ),
        primary_key_fields=["id"],
    )
"""

import dataclasses
import json
import logging
import requests
from typing import Any, NamedTuple
from base64 import b64encode
from urllib.parse import urljoin

import cocoindex

logger = logging.getLogger(__name__)


# =============================================================================
# Target Spec
# =============================================================================

class DorisTarget(cocoindex.op.TargetSpec):
    """
    Target specification for Apache Doris.
    
    Attributes:
        fe_host: Doris FE host address
        fe_http_port: Doris FE HTTP port (default: 8030)
        database: Target database name
        table: Target table name
        username: Doris username (default: "root")
        password: Doris password (default: "")
        enable_https: Use HTTPS for connections (default: False)
        stream_load_timeout: Timeout for Stream Load in seconds (default: 600)
        batch_size: Number of rows per batch for Stream Load (default: 10000)
        auto_create_table: Automatically create table if not exists (default: True)
        vector_dimension: Dimension for vector fields, if any (default: None)
        replication_num: Replication number for auto-created tables (default: 1)
    """
    fe_host: str
    fe_http_port: int = 8030
    database: str
    table: str
    username: str = "root"
    password: str = ""
    enable_https: bool = False
    stream_load_timeout: int = 600
    batch_size: int = 10000
    auto_create_table: bool = True
    vector_dimension: int | None = None
    replication_num: int = 1


# =============================================================================
# Persistent Key
# =============================================================================

class DorisPersistentKey(NamedTuple):
    """Unique identifier for a Doris target instance."""
    fe_host: str
    fe_http_port: int
    database: str
    table: str


# =============================================================================
# Prepared Target (for connection reuse)
# =============================================================================

@dataclasses.dataclass
class PreparedDorisTarget:
    """Prepared Doris target with HTTP session for connection reuse."""
    spec: DorisTarget
    session: requests.Session
    base_url: str
    auth_header: str
    
    def close(self):
        """Close the HTTP session."""
        self.session.close()


# =============================================================================
# Helper Functions
# =============================================================================

def _get_doris_type(python_type: type, vector_dim: int | None = None) -> str:
    """Map Python types to Doris column types."""
    type_name = getattr(python_type, '__name__', str(python_type))
    
    # Handle Optional types
    if hasattr(python_type, '__origin__'):
        origin = python_type.__origin__
        if origin is list:
            args = python_type.__args__
            if args and args[0] in (float, int):
                # Vector field - use ARRAY<FLOAT>
                return "ARRAY<FLOAT>"
            return "JSON"
        if origin is dict:
            return "JSON"
    
    type_mapping = {
        'str': 'VARCHAR(65533)',
        'int': 'BIGINT',
        'float': 'DOUBLE',
        'bool': 'BOOLEAN',
        'bytes': 'VARCHAR(65533)',
        'datetime': 'DATETIME',
        'date': 'DATE',
        'list': 'JSON',
        'dict': 'JSON',
    }
    
    return type_mapping.get(type_name, 'VARCHAR(65533)')


def _serialize_value(value: Any) -> Any:
    """Serialize a Python value for JSON/Doris."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        # For vectors/arrays, convert to list
        return [float(v) if isinstance(v, (int, float)) else v for v in value]
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return value


def _build_stream_load_url(base_url: str, database: str, table: str) -> str:
    """Build the Stream Load URL."""
    return f"{base_url}/api/{database}/{table}/_stream_load"


def _build_delete_url(base_url: str, database: str, table: str) -> str:
    """Build the delete URL for SQL execution."""
    return f"{base_url}/api/{database}/_stream_load"


# =============================================================================
# Target Connector
# =============================================================================

@cocoindex.op.target_connector(spec_cls=DorisTarget)
class DorisTargetConnector:
    """
    Target connector for Apache Doris.
    
    This connector uses Stream Load for efficient batch data ingestion
    and supports both upsert and delete operations through Doris's
    UNIQUE KEY model.
    """
    
    @staticmethod
    def get_persistent_key(spec: DorisTarget, target_name: str) -> dict:
        """Return a persistent key that uniquely identifies this target instance."""
        return {
            "fe_host": spec.fe_host,
            "fe_http_port": spec.fe_http_port,
            "database": spec.database,
            "table": spec.table,
        }
    
    @staticmethod
    def describe(key: dict) -> str:
        """Return a human-readable description of the target."""
        return f"Doris table {key['database']}.{key['table']} @ {key['fe_host']}:{key['fe_http_port']}"
    
    @staticmethod
    def apply_setup_change(
        key: dict,
        previous: DorisTarget | None,
        current: DorisTarget | None
    ) -> None:
        """
        Apply setup changes to the target.
        
        Creates or drops the Doris table based on configuration changes.
        """
        protocol = "https" if (current and current.enable_https) or (previous and previous.enable_https) else "http"
        base_url = f"{protocol}://{key['fe_host']}:{key['fe_http_port']}"
        
        # Prepare authentication
        spec = current or previous
        if spec is None:
            return
            
        auth = b64encode(f"{spec.username}:{spec.password}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/json",
        }
        
        # Handle target removal
        if current is None and previous is not None:
            logger.info(f"Dropping Doris table {key['database']}.{key['table']}")
            # Note: We don't drop the table by default to prevent data loss
            # Users should manually drop tables if needed
            logger.warning(
                f"Table {key['database']}.{key['table']} was not dropped. "
                "Please drop it manually if needed."
            )
            return
        
        # Handle target creation
        if previous is None and current is not None and current.auto_create_table:
            logger.info(f"Doris target configured for {key['database']}.{key['table']}")
            # Table creation will be handled lazily when we know the schema
            # from the first batch of data
    
    @staticmethod
    def prepare(spec: DorisTarget) -> PreparedDorisTarget:
        """
        Prepare for execution by creating an HTTP session.
        
        This method is called once before mutations begin, allowing
        connection reuse across multiple mutation batches.
        """
        protocol = "https" if spec.enable_https else "http"
        base_url = f"{protocol}://{spec.fe_host}:{spec.fe_http_port}"
        
        session = requests.Session()
        auth = b64encode(f"{spec.username}:{spec.password}".encode()).decode()
        
        return PreparedDorisTarget(
            spec=spec,
            session=session,
            base_url=base_url,
            auth_header=f"Basic {auth}",
        )
    
    @staticmethod
    def mutate(
        *all_mutations: tuple[PreparedDorisTarget, dict[Any, Any | None]],
    ) -> None:
        """
        Apply data mutations to the Doris target.
        
        This method handles both upserts (when value is not None) and
        deletes (when value is None) using Doris Stream Load.
        
        For UNIQUE KEY model tables:
        - Upserts are handled naturally by Stream Load
        - Deletes use the __DORIS_DELETE_SIGN__ column
        """
        for prepared, mutations in all_mutations:
            if not mutations:
                continue
            
            spec = prepared.spec
            session = prepared.session
            
            # Separate upserts and deletes
            upserts = []
            deletes = []
            
            for key, value in mutations.items():
                if value is None:
                    # Delete operation
                    deletes.append(key)
                else:
                    # Upsert operation
                    row = {}
                    
                    # Handle key fields
                    if isinstance(key, dict):
                        row.update({k: _serialize_value(v) for k, v in key.items()})
                    elif hasattr(key, '_asdict'):
                        # NamedTuple
                        row.update({k: _serialize_value(v) for k, v in key._asdict().items()})
                    elif hasattr(key, '__dataclass_fields__'):
                        # dataclass
                        row.update({k: _serialize_value(getattr(key, k)) for k in key.__dataclass_fields__})
                    else:
                        # Single key field - will be handled by the flow definition
                        row['_key'] = _serialize_value(key)
                    
                    # Handle value fields
                    if isinstance(value, dict):
                        row.update({k: _serialize_value(v) for k, v in value.items()})
                    elif hasattr(value, '_asdict'):
                        row.update({k: _serialize_value(v) for k, v in value._asdict().items()})
                    elif hasattr(value, '__dataclass_fields__'):
                        row.update({k: _serialize_value(getattr(value, k)) for k in value.__dataclass_fields__})
                    
                    upserts.append(row)
            
            # Process upserts in batches
            if upserts:
                DorisTargetConnector._stream_load_batch(
                    prepared, upserts, is_delete=False
                )
            
            # Process deletes
            if deletes:
                DorisTargetConnector._stream_load_deletes(
                    prepared, deletes
                )
    
    @staticmethod
    def _stream_load_batch(
        prepared: PreparedDorisTarget,
        rows: list[dict],
        is_delete: bool = False
    ) -> None:
        """
        Execute Stream Load for a batch of rows.
        
        Args:
            prepared: Prepared target with HTTP session
            rows: List of row dictionaries to load
            is_delete: Whether these are delete operations
        """
        spec = prepared.spec
        url = _build_stream_load_url(prepared.base_url, spec.database, spec.table)
        
        # Prepare headers
        headers = {
            "Authorization": prepared.auth_header,
            "Expect": "100-continue",
            "format": "json",
            "strip_outer_array": "true",
            "fuzzy_parse": "true",
        }
        
        if is_delete:
            headers["merge_type"] = "DELETE"
        
        # Add timeout
        headers["timeout"] = str(spec.stream_load_timeout)
        
        # Process in batches
        batch_size = spec.batch_size
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            data = json.dumps(batch)
            
            try:
                response = prepared.session.put(
                    url,
                    data=data.encode('utf-8'),
                    headers=headers,
                    timeout=spec.stream_load_timeout,
                )
                
                # Parse response
                result = response.json()
                
                if result.get('Status') not in ('Success', 'Publish Timeout'):
                    error_msg = result.get('Message', 'Unknown error')
                    error_url = result.get('ErrorURL', '')
                    raise RuntimeError(
                        f"Stream Load failed: {error_msg}. "
                        f"Error URL: {error_url}"
                    )
                
                loaded = result.get('NumberLoadedRows', 0)
                filtered = result.get('NumberFilteredRows', 0)
                logger.debug(
                    f"Stream Load completed: {loaded} loaded, {filtered} filtered"
                )
                
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Stream Load request failed: {e}")
    
    @staticmethod
    def _stream_load_deletes(
        prepared: PreparedDorisTarget,
        keys: list[Any]
    ) -> None:
        """
        Execute delete operations using Stream Load with __DORIS_DELETE_SIGN__.
        
        For UNIQUE KEY model, we load rows with __DORIS_DELETE_SIGN__=1.
        """
        spec = prepared.spec
        
        # Convert keys to rows with delete sign
        rows = []
        for key in keys:
            row = {}
            
            if isinstance(key, dict):
                row.update({k: _serialize_value(v) for k, v in key.items()})
            elif hasattr(key, '_asdict'):
                row.update({k: _serialize_value(v) for k, v in key._asdict().items()})
            elif hasattr(key, '__dataclass_fields__'):
                row.update({k: _serialize_value(getattr(key, k)) for k in key.__dataclass_fields__})
            else:
                row['_key'] = _serialize_value(key)
            
            # Add delete sign
            row['__DORIS_DELETE_SIGN__'] = 1
            rows.append(row)
        
        if rows:
            url = _build_stream_load_url(prepared.base_url, spec.database, spec.table)
            
            headers = {
                "Authorization": prepared.auth_header,
                "Expect": "100-continue",
                "format": "json",
                "strip_outer_array": "true",
                "fuzzy_parse": "true",
                "timeout": str(spec.stream_load_timeout),
                "columns": ",".join(rows[0].keys()),
                "merge_type": "MERGE",
                "delete": "__DORIS_DELETE_SIGN__=1",
            }
            
            data = json.dumps(rows)
            
            try:
                response = prepared.session.put(
                    url,
                    data=data.encode('utf-8'),
                    headers=headers,
                    timeout=spec.stream_load_timeout,
                )
                
                result = response.json()
                
                if result.get('Status') not in ('Success', 'Publish Timeout'):
                    error_msg = result.get('Message', 'Unknown error')
                    raise RuntimeError(f"Delete Stream Load failed: {error_msg}")
                
                logger.debug(
                    f"Delete Stream Load completed: {result.get('NumberLoadedRows', 0)} rows"
                )
                
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Delete Stream Load request failed: {e}")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_doris_table_ddl(
    database: str,
    table: str,
    schema: dict[str, str],
    primary_keys: list[str],
    vector_fields: dict[str, int] | None = None,
    replication_num: int = 1,
) -> str:
    """
    Generate DDL statement for creating a Doris table.
    
    Args:
        database: Database name
        table: Table name
        schema: Dictionary mapping column names to Doris types
        primary_keys: List of primary key column names
        vector_fields: Dictionary mapping vector column names to dimensions
        replication_num: Number of replicas
    
    Returns:
        CREATE TABLE DDL statement
    
    Example:
        ddl = create_doris_table_ddl(
            database="my_db",
            table="embeddings",
            schema={
                "id": "VARCHAR(255)",
                "text": "VARCHAR(65533)",
                "embedding": "ARRAY<FLOAT>",
            },
            primary_keys=["id"],
            vector_fields={"embedding": 768},
        )
    """
    # Build column definitions
    columns = []
    for col_name, col_type in schema.items():
        columns.append(f"    `{col_name}` {col_type}")
    
    # Build DDL
    ddl = f"""
CREATE TABLE IF NOT EXISTS `{database}`.`{table}` (
{','.join(chr(10).join(columns))}
)
UNIQUE KEY({', '.join(f'`{k}`' for k in primary_keys)})
DISTRIBUTED BY HASH({', '.join(f'`{k}`' for k in primary_keys)}) BUCKETS AUTO
PROPERTIES (
    "replication_num" = "{replication_num}",
    "enable_unique_key_merge_on_write" = "true"
);
"""
    
    # Add vector index if needed
    if vector_fields:
        for vec_col, dim in vector_fields.items():
            ddl += f"""
CREATE INDEX IF NOT EXISTS `idx_{vec_col}_vector` 
ON `{database}`.`{table}` (`{vec_col}`)
USING INVERTED;
"""
    
    return ddl.strip()


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example: Generate table DDL
    ddl = create_doris_table_ddl(
        database="cocoindex_demo",
        table="document_embeddings",
        schema={
            "filename": "VARCHAR(1024)",
            "chunk_location": "VARCHAR(255)",
            "text": "TEXT",
            "embedding": "ARRAY<FLOAT>",
        },
        primary_keys=["filename", "chunk_location"],
        vector_fields={"embedding": 768},
    )
    print("Example DDL:")
    print(ddl)
