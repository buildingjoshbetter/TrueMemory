"""
TrueMemory Cloud Proxy
======================

Routes MCP tool calls to TrueMemory Cloud instead of the local SQLite database.
Implements the same interface surface as the local engine so ``mcp_server.py``
can swap between local and cloud mode transparently.

Cloud mode is opt-in via ``~/.truememory/config.json``::

    {
      "mode": "cloud",
      "cloud_api_url": "https://api.truememory.net",
      "cloud_api_key": "tm_live_..."
    }
"""

from __future__ import annotations

import logging
import time
from urllib.parse import quote as urlquote

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class CloudProxyError(Exception):
    """Raised when a cloud API call fails after retries."""


class CloudProxy:
    """Proxy that routes TrueMemory operations to the cloud API.

    Parameters
    ----------
    api_url : str
        Base URL of the TrueMemory Cloud API (no trailing slash).
    api_key : str
        Bearer token for authentication.
    timeout : float
        Per-request timeout in seconds (default 30).
    """

    def __init__(self, api_url: str, api_key: str, timeout: float = 30.0):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.Client(
            base_url=self.api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "truememory-mcp/cloud-proxy",
            },
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
    ) -> dict | list:
        """Execute an HTTP request with retry logic for 429/5xx errors.

        Returns the parsed JSON response body on success.
        Raises ``CloudProxyError`` with a human-readable message on failure.
        """
        last_exc: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.request(
                    method,
                    path,
                    json=json_body,
                    params=params,
                )

                if response.status_code in _RETRYABLE_STATUS_CODES:
                    # Respect Retry-After header if present
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = float(retry_after)
                        except ValueError:
                            wait = _RETRY_BACKOFF_BASE * (2 ** attempt)
                    else:
                        wait = _RETRY_BACKOFF_BASE * (2 ** attempt)

                    log.warning(
                        "Cloud API %s %s returned %d, retrying in %.1fs (attempt %d/%d)",
                        method, path, response.status_code, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                if response.status_code == 401:
                    raise CloudProxyError(
                        "Authentication failed. Check your cloud_api_key in "
                        "~/.truememory/config.json or re-run truememory_configure."
                    )
                if response.status_code == 403:
                    raise CloudProxyError(
                        "Access denied. Your API key may lack the required permissions."
                    )
                if response.status_code == 404:
                    raise CloudProxyError(
                        f"Resource not found: {method} {path}"
                    )
                if response.status_code == 422:
                    # Validation error -- return the detail as-is
                    try:
                        detail = response.json()
                    except Exception:
                        detail = {"error": response.text}
                    raise CloudProxyError(
                        f"Validation error: {detail}"
                    )

                # Any other non-2xx
                response.raise_for_status()

                # Success
                return response.json()

            except CloudProxyError:
                raise
            except httpx.TimeoutException as e:
                last_exc = e
                wait = _RETRY_BACKOFF_BASE * (2 ** attempt)
                log.warning(
                    "Cloud API %s %s timed out, retrying in %.1fs (attempt %d/%d)",
                    method, path, wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)
            except httpx.HTTPStatusError as e:
                # Non-retryable HTTP error
                raise CloudProxyError(
                    f"Cloud API error: {e.response.status_code} {e.response.reason_phrase} "
                    f"for {method} {path}"
                ) from e
            except httpx.HTTPError as e:
                last_exc = e
                wait = _RETRY_BACKOFF_BASE * (2 ** attempt)
                log.warning(
                    "Cloud API %s %s connection error (%s), retrying in %.1fs (attempt %d/%d)",
                    method, path, type(e).__name__, wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)

        raise CloudProxyError(
            f"Cloud API {method} {path} failed after {_MAX_RETRIES} retries: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Public interface — mirrors the local engine operations
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        user_id: str = "",
        metadata: dict | None = None,
    ) -> dict:
        """Store a memory via the cloud API.

        POST /v1/memories
        """
        body: dict = {"content": content}
        if user_id:
            body["user_id"] = user_id
        if metadata:
            body["metadata"] = metadata
        return self._request("POST", "/v1/memories", json_body=body)

    def search(
        self,
        query: str,
        user_id: str = "",
        limit: int = 10,
        queries: list[str] | None = None,
    ) -> list[dict]:
        """Search memories via the cloud API.

        POST /v1/memories/search
        """
        body: dict = {"query": query, "limit": limit}
        if user_id:
            body["user_id"] = user_id
        if queries:
            body["queries"] = queries
        result = self._request("POST", "/v1/memories/search", json_body=body)
        return result if isinstance(result, list) else result.get("results", [])

    def search_deep(
        self,
        query: str,
        user_id: str = "",
        limit: int = 10,
        queries: list[str] | None = None,
    ) -> list[dict]:
        """Deep search memories via the cloud API.

        POST /v1/memories/search/deep
        """
        body: dict = {"query": query, "limit": limit}
        if user_id:
            body["user_id"] = user_id
        if queries:
            body["queries"] = queries
        result = self._request("POST", "/v1/memories/search/deep", json_body=body)
        return result if isinstance(result, list) else result.get("results", [])

    def get(self, memory_id: int) -> dict | None:
        """Get a specific memory by ID.

        GET /v1/memories/{memory_id}
        """
        try:
            return self._request("GET", f"/v1/memories/{memory_id}")
        except CloudProxyError as e:
            if "not found" in str(e).lower():
                return None
            raise

    def forget(self, memory_id: int) -> bool:
        """Delete a memory by ID.

        DELETE /v1/memories/{memory_id}
        """
        try:
            self._request("DELETE", f"/v1/memories/{memory_id}")
            return True
        except CloudProxyError as e:
            if "not found" in str(e).lower():
                return False
            raise

    def stats(self) -> dict:
        """Get memory system statistics.

        GET /v1/stats
        """
        return self._request("GET", "/v1/stats")

    def configure(self, tier: str | None = None, **settings) -> dict:
        """Update cloud configuration.

        POST /v1/configure
        """
        body: dict = dict(settings)
        if tier:
            body["tier"] = tier
        return self._request("POST", "/v1/configure", json_body=body)

    def entity_profile(self, entity: str) -> dict:
        """Get entity personality profile.

        GET /v1/entities/{entity}
        """
        return self._request("GET", f"/v1/entities/{urlquote(entity, safe='')}")

    def status(self, status_id: int = 0) -> dict:
        """Check rebuild/re-embedding status.

        GET /v1/status/{status_id}
        """
        return self._request("GET", f"/v1/status/{status_id}")

    def close(self) -> None:
        """Close the underlying httpx client."""
        try:
            self._client.close()
        except Exception:
            pass
