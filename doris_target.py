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
import time
import logging
import os
import requests
from requests.auth import HTTPBasicAuth
import uuid
from typing import cast

# Optional NumPy support for serialization
try:
    import numpy as _np  # type: ignore
except Exception:  # numpy not required
    _np = None
from typing import Any
from base64 import b64encode
from doris_http import (
    sanitize_headers_for_log,
    put_with_manual_redirect,
)

import cocoindex

logger = logging.getLogger(__name__)
# Attach a default console handler if none configured by host app
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(_h)
# Allow environment to control log level; default INFO
_lvl = os.getenv("DORIS_LOG_LEVEL", "INFO").upper()
logger.setLevel(getattr(logging, _lvl, logging.INFO))


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
    database: str
    table: str
    fe_http_port: int = 8030
    username: str = "root"
    password: str = ""
    enable_https: bool = False
    stream_load_timeout: int = 600
    batch_size: int = 10000
    auto_create_table: bool = True
    vector_dimension: int | None = None
    replication_num: int = 1
    # MySQL query port for executing DDL (default Doris: 9030)
    query_port: int = 9030


# =============================================================================
# Persistent Key
# =============================================================================

# Removed unused DorisPersistentKey


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
    # Convert UUIDs to strings for JSON
    if isinstance(value, uuid.UUID):
        return str(value)
    # Normalize NaN floats to None
    if isinstance(value, float) and value != value:
        return None
    # Handle numpy scalar types
    if _np is not None and isinstance(value, (_np.floating, _np.integer)):
        py_val = cast(Any, value).item()
        if isinstance(py_val, float) and py_val != py_val:
            return None
        return py_val
    # Handle numpy arrays
    if _np is not None and isinstance(value, _np.ndarray):
        return [_serialize_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        # Recursively serialize list/tuple elements
        return [_serialize_value(v) for v in value]
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


def _infer_doris_type_from_value(value: Any) -> str:
    """Infer a Doris column type from a Python value."""
    if value is None:
        return 'VARCHAR(65533)'
    if isinstance(value, uuid.UUID):
        return 'VARCHAR(255)'
    if isinstance(value, bool):
        return 'BOOLEAN'
    if isinstance(value, int):
        return 'BIGINT'
    if isinstance(value, float):
        return 'DOUBLE'
    try:
        import numpy as _np_local  # type: ignore
        if isinstance(value, (_np_local.integer,)):
            return 'BIGINT'
        if isinstance(value, (_np_local.floating,)):
            return 'DOUBLE'
    except Exception:
        pass
    if isinstance(value, (bytes, bytearray)):
        return 'VARCHAR(65533)'
    if isinstance(value, str):
        return 'VARCHAR(65533)'
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 'JSON'
        def _is_num(x: Any) -> bool:
            if isinstance(x, (int, float)):
                return True
            try:
                import numpy as _np_local2  # type: ignore
                if isinstance(x, (_np_local2.integer, _np_local2.floating)):
                    return True
            except Exception:
                pass
            return False
        if all(_is_num(x) for x in value):
            return 'ARRAY<FLOAT>'
        return 'JSON'
    if isinstance(value, dict):
        return 'JSON'
    if hasattr(value, 'isoformat'):
        try:
            value.isoformat()
            return 'DATETIME'
        except Exception:
            return 'VARCHAR(65533)'
    return 'VARCHAR(65533)'


def _infer_schema_from_rows(rows: list[dict]) -> dict[str, str]:
    """Infer a Doris schema mapping (col -> type) from sample rows."""
    schema: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            t = _infer_doris_type_from_value(v)
            prev = schema.get(k)
            if prev is None:
                schema[k] = t
            elif prev != t:
                numeric_types = {'BIGINT', 'DOUBLE'}
                if (prev in numeric_types and t in numeric_types):
                    schema[k] = 'DOUBLE'
                else:
                    schema[k] = 'JSON'
    return schema


def _guess_primary_keys(rows: list[dict]) -> list[str]:
    """Guess primary keys from sample rows."""
    if not rows:
        return ['id']
    r = rows[0]
    if 'id' in r:
        return ['id']
    # Prefer only filename as key/distribution column
    if 'filename' in r:
        return ['filename']
    if '_key' in r:
        return ['_key']
    # Fallback: first non-vector column
    for k, v in r.items():
        t = _infer_doris_type_from_value(v)
        if t != 'ARRAY<FLOAT>' and t != 'JSON':
            return [k]
    return ['id']


def _guess_vector_fields(rows: list[dict]) -> dict[str, int]:
    vf: dict[str, int] = {}
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, (int, float)) for x in v):
                vf[k] = len(v)
    return vf



def _ensure_table_exists(prepared: PreparedDorisTarget, sample_rows: list[dict]) -> None:
    spec = prepared.spec
    if not spec.auto_create_table or not sample_rows:
        return
    schema = _infer_schema_from_rows(sample_rows)
    pks = _guess_primary_keys(sample_rows)
    vectors = _guess_vector_fields(sample_rows)
    ddl = create_doris_table_ddl(
        database=spec.database,
        table=spec.table,
        schema=schema,
        primary_keys=pks,
        vector_fields=vectors if vectors else None,
        replication_num=spec.replication_num,
    )
    logger.info(f"Ensuring table exists with DDL:\n{ddl}")
    # Use MySQL connector only; fail fast on errors
    import mysql.connector  # type: ignore
    conn = mysql.connector.connect(
        host=spec.fe_host,
        port=spec.query_port,
        user=spec.username,
        password=spec.password,
    )
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{spec.database}`;")
    # Execute DDL statements sequentially to avoid multi=True generator issues
    statements = [s.strip() for s in ddl.split(';') if s.strip()]
    for stmt in statements:
        cur.execute(stmt)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Table ensured via MySQL connector")
# Removed local header sanitizer (using shared sanitize_headers_for_log)


# Removed local redirect PUT (using shared put_with_manual_redirect)

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
        # Avoid inheriting proxy/env that might strip or alter Authorization
        session.trust_env = False
        auth = b64encode(f"{spec.username}:{spec.password}".encode()).decode()
        logger.info(
            f"Preparing Doris target: {spec.database}.{spec.table} @ {spec.fe_host}:{spec.fe_http_port}"
        )
        # Default auth for all requests
        session.auth = HTTPBasicAuth(spec.username, spec.password)
        
        prepared = PreparedDorisTarget(
            spec=spec,
            session=session,
            base_url=base_url,
            auth_header=f"Basic {auth}",
        )
        return prepared
    
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
            
            logger.info(
                f"Mutating Doris target {spec.database}.{spec.table}: upserts={len(upserts)} deletes={len(deletes)}"
            )

            # Ensure table exists before first upsert (fail fast)
            if upserts and spec.auto_create_table:
                _ensure_table_exists(prepared, upserts[: min(len(upserts), spec.batch_size)])

            # Process upserts in batches using Stream Load
            if upserts:
                DorisTargetConnector._stream_load_batch(prepared, upserts, is_delete=False)
            
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
        logger.info(
            f"Stream Load to {url} with table: {spec.database}.{spec.table}: rows={len(rows)} delete={is_delete} batch_size={spec.batch_size}"
        )
        # Prepare headers (align with doris_vector_search: send Basic Authorization header)
        headers = {
            "Expect": "100-continue",
            "format": "json",
            "strip_outer_array": "true",
            "fuzzy_parse": "true",
            "Authorization": prepared.auth_header,
            "Content-Type": "application/json; charset=utf-8",
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
            logger.debug(
                f"Uploading batch {i // batch_size + 1} size={len(batch)} to {url}"
            )
            # Use a unique label per batch to aid FE diagnostics
            batch_headers = dict(headers)
            batch_headers["label"] = f"cocoindex_{int(time.time()*1000)}_{(i // batch_size) + 1}"
            
            try:
                response = put_with_manual_redirect(
                    prepared.session,
                    url,
                    batch_headers,
                    data.encode('utf-8'),
                    spec.stream_load_timeout,
                )
                
                # Parse response
                text = response.text or ""
                try:
                    result = response.json()
                except ValueError:
                    result = None
                if result is None:
                    logger.error(
                        f"Stream Load non-JSON response: HTTP {response.status_code} {response.reason} url={response.url}"
                    )
                    logger.error(
                        f"Response headers: {json.dumps(sanitize_headers_for_log(dict(response.headers)), ensure_ascii=False)}"
                    )
                    body_preview = text if len(text) <= 4000 else text[:4000] + "... [truncated]"
                    logger.error(f"Response body: {body_preview}")
                    raise RuntimeError("Stream Load failed: non-JSON response from FE")
                else:
                    try:
                        logger.info(f"Stream Load raw result: {json.dumps(result, ensure_ascii=False)}")
                    except Exception:
                        logger.info(f"Stream Load raw result (repr): {result!r}")
                
                status_val = None
                if isinstance(result, dict):
                    status_val = result.get('Status') or result.get('status')
                status_ok = False
                if isinstance(status_val, str):
                    status_ok = status_val in ("Success", "Publish Timeout") or status_val.lower() in ("success", "publish timeout", "ok")
                if not status_ok:
                    error_msg = result.get('Message') or result.get('msg') or 'Unknown error'
                    error_url = result.get('ErrorURL', '')
                    logger.error(
                        f"Stream Load failed: {error_msg}. Error URL: {error_url}"
                    )
                    raise RuntimeError(
                        f"Stream Load failed: {error_msg}. "
                        f"Error URL: {error_url}"
                    )
                
                loaded = result.get('NumberLoadedRows', result.get('numberLoadedRows', 0))
                filtered = result.get('NumberFilteredRows', result.get('numberFilteredRows', 0))
                logger.debug(
                    f"Stream Load completed: {loaded} loaded, {filtered} filtered"
                )
                
            except requests.exceptions.RequestException as e:
                logger.exception("Stream Load request failed")
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
            logger.info(
                f"Stream Load deletes to {spec.database}.{spec.table}: rows={len(rows)}"
            )
            
            headers = {
                "Expect": "100-continue",
                "format": "json",
                "strip_outer_array": "true",
                "fuzzy_parse": "true",
                "timeout": str(spec.stream_load_timeout),
                "columns": ",".join(rows[0].keys()),
                "merge_type": "MERGE",
                "delete": "__DORIS_DELETE_SIGN__=1",
                "Authorization": prepared.auth_header,
                "Content-Type": "application/json; charset=utf-8",
            }
            
            data = json.dumps(rows)
            
            try:
                response = put_with_manual_redirect(
                    prepared.session,
                    url,
                    headers,
                    data.encode('utf-8'),
                    spec.stream_load_timeout,
                )
                
                text = response.text or ""
                try:
                    result = response.json()
                except ValueError:
                    result = None
                if result is None:
                    logger.error(
                        f"Delete Stream Load non-JSON response: HTTP {response.status_code} {response.reason} url={response.url}"
                    )
                    logger.error(
                        f"Response headers: {json.dumps(sanitize_headers_for_log(dict(response.headers)), ensure_ascii=False)}"
                    )
                    body_preview = text if len(text) <= 4000 else text[:4000] + "... [truncated]"
                    logger.error(f"Response body: {body_preview}")
                    raise RuntimeError("Delete Stream Load failed: non-JSON response from FE")
                else:
                    try:
                        logger.info(f"Delete Stream Load raw result: {json.dumps(result, ensure_ascii=False)}")
                    except Exception:
                        logger.info(f"Delete Stream Load raw result (repr): {result!r}")
                
                status_val = None
                if isinstance(result, dict):
                    status_val = result.get('Status') or result.get('status')
                status_ok = False
                if isinstance(status_val, str):
                    status_ok = status_val in ("Success", "Publish Timeout") or status_val.lower() in ("success", "publish timeout", "ok")
                if not status_ok:
                    error_msg = result.get('Message') or result.get('msg') or 'Unknown error'
                    logger.error(f"Delete Stream Load failed: {error_msg}")
                    raise RuntimeError(f"Delete Stream Load failed: {error_msg}")
                
                logger.debug(
                    f"Delete Stream Load completed: {result.get('NumberLoadedRows', 0)} rows"
                )
                
            except requests.exceptions.RequestException as e:
                logger.exception("Delete Stream Load request failed")
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
    Generate DDL statement for creating a Doris table with ANN index (Doris 4.x).

    Args:
        database: Database name
        table: Table name
        schema: Dictionary mapping column names to Doris types
        primary_keys: List of key column names used for DUPLICATE KEY
        vector_fields: Dictionary mapping vector column names to dimensions
        replication_num: Number of replicas

    Returns:
        CREATE TABLE DDL statement using ARRAY<FLOAT> and USING ANN index.

    Notes:
        - Doris 4.x ANN requires vector columns as NOT NULL ARRAY<FLOAT>.
        - ANN is supported only on DUPLICATE KEY model tables.
        - For DUPLICATE KEY, key columns MUST be an ordered prefix of the schema.
        - Index PROPERTIES include index_type=hnsw, metric_type=l2_distance, dim.
    """
    # Build column order: primary keys first (ordered), then the rest in original order
    vec_fields = vector_fields or {}

    # Filter out unsupported key columns (ARRAY/JSON) and ensure at least one key
    def _is_scalar_type(t: str) -> bool:
        tt = (t or '').upper()
        return ('ARRAY<' not in tt) and (tt != 'JSON')

    filtered_pks = [k for k in primary_keys if _is_scalar_type(schema.get(k, 'VARCHAR(65533)'))]
    if not filtered_pks:
        # Prefer filename when present and scalar
        if 'filename' in schema and _is_scalar_type(schema.get('filename', 'VARCHAR(65533)')):
            filtered_pks = ['filename']
        else:
            # Fallback to first scalar column in schema order
            for col, col_t in schema.items():
                if _is_scalar_type(col_t):
                    filtered_pks = [col]
                    break
            if not filtered_pks:
                filtered_pks = ['id']

    pk_set = set(filtered_pks)
    # Original order from inferred schema
    original_cols = list(schema.keys())
    # Ensure all PKs exist in schema; if missing, add as VARCHAR
    for pk in filtered_pks:
        if pk not in schema:
            schema[pk] = 'VARCHAR(65533)'
            if pk not in original_cols:
                original_cols.insert(0, pk)
    ordered_cols: list[str] = []
    # Add PKs in order
    for pk in filtered_pks:
        if pk in ordered_cols:
            continue
        ordered_cols.append(pk)
    # Add the rest
    for col in original_cols:
        if col in pk_set:
            continue
        ordered_cols.append(col)

    # Build column definitions with ordered columns
    columns: list[str] = []
    for col_name in ordered_cols:
        col_type = schema[col_name]
        # Ensure vector columns are NOT NULL ARRAY<FLOAT>
        if col_name in vec_fields:
            base_type = "ARRAY<FLOAT>"
            not_null = "NOT NULL" if "NOT NULL" not in col_type.upper() else ""
            columns.append(f"    `{col_name}` {base_type} {not_null}".rstrip())
        else:
            columns.append(f"    `{col_name}` {col_type}")
    
    # Build DDL
    ddl = f"""
CREATE TABLE IF NOT EXISTS `{database}`.`{table}` (
    {',\n'.join(columns)}
)
DUPLICATE KEY({', '.join(f'`{k}`' for k in filtered_pks)})
DISTRIBUTED BY HASH({', '.join(f'`{k}`' for k in filtered_pks)}) BUCKETS AUTO
PROPERTIES (
    "replication_num" = "{replication_num}"
);
"""
    
    # Add ANN index inline statements after table DDL (Doris also supports inline within CREATE TABLE,
    # but executing separate CREATE INDEX is acceptable and clearer here.)
    if vec_fields:
        for vec_col, dim in vec_fields.items():
            # Default to HNSW + L2; quantizer flat
            ddl += f"""
CREATE INDEX IF NOT EXISTS `idx_{vec_col}_ann`
ON `{database}`.`{table}` (`{vec_col}`)
USING ANN PROPERTIES (
    "index_type" = "hnsw",
    "metric_type" = "l2_distance",
    "dim" = "{dim}",
    "quantizer" = "flat"
);
"""
    
    # Add inverted index for TEXT column named 'text'
    text_col_type = schema.get('text')
    if text_col_type and 'TEXT' in str(text_col_type).upper():
        ddl += f"""
CREATE INDEX IF NOT EXISTS `idx_text_inverted`
ON `{database}`.`{table}` (`text`)
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
