import json
from typing import Dict, Optional
from urllib.parse import urljoin
import requests
import logging

logger = logging.getLogger(__name__)


def sanitize_headers_for_log(headers: Dict[str, str]) -> Dict[str, str]:
    """Return a sanitized copy of headers suitable for logging."""
    redacted: Dict[str, str] = {}
    for k, v in headers.items():
        kl = k.lower()
        sv = str(v)
        if kl in ("authorization",):
            redacted[k] = "<redacted>"
        else:
            redacted[k] = sv if len(sv) < 512 else sv[:512] + "..."
    return redacted


def put_with_manual_redirect(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    data: bytes,
    timeout: int,
) -> requests.Response:
    """Perform PUT and manually follow a single redirect to preserve Authorization."""
    resp = session.put(
        url,
        data=data,
        headers=headers,
        allow_redirects=False,
        timeout=timeout,
    )
    if resp.status_code in (301, 302, 303, 307, 308):
        loc = resp.headers.get("Location")
        if loc:
            redirect_url = urljoin(resp.url, loc)
            logger.info(f"Following redirect to {redirect_url}")
            resp = session.put(
                redirect_url,
                data=data,
                headers=headers,
                allow_redirects=False,
                timeout=timeout,
            )
    return resp


def post_sql(
    session: requests.Session,
    base_url: str,
    auth_header: str,
    sql: str,
    database: Optional[str],
    timeout: int,
) -> requests.Response:
    """Attempt to execute SQL via Doris HTTP endpoints, trying multiple paths.

    Tries in order:
      1) POST /api/_sql with body {"stmt": sql}
      2) POST /api/sql with body {"sql": sql, "database": database}
      3) POST /rest/v2/sql with body {"sql": sql, "database": database}
    Returns the final response.
    """
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
    }

    # 1) /api/_sql with 'stmt'
    url1 = f"{base_url}/api/_sql"
    body1 = {"stmt": sql}
    r1 = session.post(url1, json=body1, headers=headers, allow_redirects=False, timeout=timeout)
    if r1.status_code < 400:
        return r1
    logger.warning(
        f"HTTP SQL (stmt) failed: {r1.status_code} {r1.text}. Retrying with 'sql'..."
    )

    # 2) /api/sql with 'sql'
    url2 = f"{base_url}/api/sql"
    body2 = {"sql": sql}
    if database:
        body2["database"] = database
    r2 = session.post(url2, json=body2, headers=headers, allow_redirects=False, timeout=timeout)
    if r2.status_code < 400:
        return r2
    logger.error(
        f"HTTP SQL (/api/sql) failed: {r2.status_code} {r2.text}. Retrying with /rest/v2/sql..."
    )

    # 3) /rest/v2/sql
    url3 = f"{base_url}/rest/v2/sql"
    body3 = {"sql": sql}
    if database:
        body3["database"] = database
    r3 = session.post(url3, json=body3, headers=headers, allow_redirects=False, timeout=timeout)
    return r3
