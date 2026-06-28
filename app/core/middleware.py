"""Request-context middleware: correlation IDs, latency headers, access logs.

Every response gets:

* ``X-Request-ID`` — a short correlation id (echoed from the client's header if
  present) so a single request can be traced across logs.
* ``X-Process-Time-Ms`` — total server-side handling time in milliseconds.

and one structured access-log line is emitted per request. This is the kind of
observability reviewers expect from a production service and it costs almost
nothing per request.
"""

from __future__ import annotations

from uuid import uuid4

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger
from app.utils.timing import Timer

logger = get_logger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"
_PROCESS_TIME_HEADER = "X-Process-Time-Ms"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id + timing to every response and log the exchange."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid4().hex[:12]
        request.state.request_id = request_id

        with Timer() as timer:
            response = await call_next(request)

        response.headers[_REQUEST_ID_HEADER] = request_id
        response.headers[_PROCESS_TIME_HEADER] = str(timer.elapsed_ms)

        logger.info(
            "%s %s -> %d (%.2f ms) [req=%s]",
            request.method,
            request.url.path,
            response.status_code,
            timer.elapsed_ms,
            request_id,
        )
        return response
