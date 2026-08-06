from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from .constants import (
    HTTP_DEFAULT_PORT,
    HTTPS_DEFAULT_PORT,
    WEB_CONTENT_TYPE_SAMPLE_BYTES,
    WEB_URL_MAX_CHARS,
)
from .storage import Storage

if TYPE_CHECKING:
    from .config import Settings


URL_PATTERN = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
ALLOWED_CONTENT_TYPES = {
    "application/json",
    "application/xhtml+xml",
    "text/html",
    "text/plain",
}
BLOCKED_HOST_SUFFIXES = (
    ".internal",
    ".local",
    ".localhost",
    ".home.arpa",
)
BLOCKED_HOSTS = {
    "host.docker.internal",
    "gateway.docker.internal",
    "kubernetes.docker.internal",
    "localhost",
}


class WebSourceError(RuntimeError):
    pass


class UnsafeUrlError(WebSourceError):
    pass


class PageFetchError(WebSourceError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        attempt_count: int = 1,
        reason: str = "fetch_failed",
        search_fallback_eligible: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.attempt_count = max(1, int(attempt_count))
        self.reason = reason
        self.search_fallback_eligible = bool(search_fallback_eligible)


@dataclass(frozen=True)
class FetchedPage:
    requested_url: str
    final_url: str
    title: str
    content_text: str
    content_type: str
    byte_count: int
    truncated: bool
    content_sha256: str
    retrieval_method: str = "direct_http"
    retrieval_attempts: int = 1


class ReadableHTMLParser(HTMLParser):
    _hidden_tags = {"canvas", "noscript", "script", "style", "svg", "template"}
    _block_tags = {
        "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
        "figcaption", "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "hr", "li", "main", "nav", "ol", "p", "pre", "section",
        "table", "tbody", "td", "th", "thead", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._hidden_depth = 0
        self._in_title = False
        self._parts: list[str] = []
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in self._hidden_tags:
            self._hidden_depth += 1
            return
        if self._hidden_depth:
            return
        if tag == "title":
            self._in_title = True
        if tag in self._block_tags:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._hidden_tags:
            self._hidden_depth = max(0, self._hidden_depth - 1)
            return
        if self._hidden_depth:
            return
        if tag == "title":
            self._in_title = False
        if tag in self._block_tags:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._hidden_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        else:
            self._parts.append(data)

    @property
    def title(self) -> str:
        return " ".join(" ".join(self._title_parts).split())

    @property
    def text(self) -> str:
        lines: list[str] = []
        for raw_line in "".join(self._parts).splitlines():
            line = " ".join(raw_line.split())
            if line and (not lines or line != lines[-1]):
                lines.append(line)
        return "\n".join(lines)


def _trim_url_punctuation(candidate: str) -> str:
    trimmed = candidate.rstrip(".,;:!?")
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    changed = True
    while trimmed and changed:
        changed = False
        for opening, closing in pairs:
            if trimmed.endswith(closing) and trimmed.count(opening) < trimmed.count(closing):
                trimmed = trimmed[:-1]
                changed = True
    return trimmed


def normalize_public_url(url: str) -> str:
    candidate = _trim_url_punctuation(str(url).strip())
    if not candidate or len(candidate) > WEB_URL_MAX_CHARS:
        raise UnsafeUrlError("The URL is empty or exceeds the supported length.")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise UnsafeUrlError("The URL is malformed.") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise UnsafeUrlError("Only http and https URLs are allowed.")
    if parsed.username is not None or parsed.password is not None:
        raise UnsafeUrlError("URLs containing credentials are not allowed.")
    hostname = (parsed.hostname or "").rstrip(".").lower()
    if not hostname:
        raise UnsafeUrlError("The URL must include a hostname.")
    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise UnsafeUrlError("The URL hostname is invalid.") from exc
    default_port = HTTPS_DEFAULT_PORT if scheme == "https" else HTTP_DEFAULT_PORT
    if port not in {None, default_port}:
        raise UnsafeUrlError("Only standard web ports 80 and 443 are allowed.")
    if hostname in BLOCKED_HOSTS or any(hostname.endswith(suffix) for suffix in BLOCKED_HOST_SUFFIXES):
        raise UnsafeUrlError("Local and private hostnames cannot be fetched.")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError as exc:
        if "." not in hostname:
            raise UnsafeUrlError("Single-label hostnames cannot be fetched.") from exc
    else:
        if not address.is_global:
            raise UnsafeUrlError("Local, private, reserved, and link-local addresses are blocked.")
    host_display = f"[{hostname}]" if ":" in hostname else hostname
    netloc = host_display if port in {None, default_port} else f"{host_display}:{port}"
    return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))


def extract_prompt_urls(message: str, max_urls: int = 3) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in URL_PATTERN.finditer(message):
        try:
            normalized = normalize_public_url(match.group(0))
        except UnsafeUrlError:
            normalized = _trim_url_punctuation(match.group(0))
        if normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
        if len(urls) >= max(1, max_urls):
            break
    return urls


class WebPageFetcher:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: Callable[[str, int], Awaitable[list[str]]] | None = None,
    ) -> None:
        self.settings = settings
        self.transport = transport
        self.resolver = resolver or self._resolve_addresses

    async def fetch(self, requested_url: str) -> FetchedPage:
        requested = normalize_public_url(requested_url)
        timeout = httpx.Timeout(
            self.settings.web_fetch_timeout_seconds,
            connect=self.settings.web_fetch_connect_timeout_seconds,
        )
        profiles = self.settings.web_fetch_user_agents[
            : self.settings.web_fetch_user_agent_attempts
        ]
        for attempt_index, user_agent in enumerate(profiles, start=1):
            try:
                return await self._fetch_once(
                    requested,
                    timeout=timeout,
                    user_agent=user_agent,
                    attempt_count=attempt_index,
                )
            except PageFetchError as exc:
                exc.attempt_count = attempt_index
                if (
                    exc.status_code not in {401, 403, 429, 451}
                    or attempt_index >= len(profiles)
                ):
                    raise
        raise PageFetchError("The page fetch attempts ended unexpectedly.")

    async def _fetch_once(
        self,
        requested: str,
        *,
        timeout: httpx.Timeout,
        user_agent: str,
        attempt_count: int,
    ) -> FetchedPage:
        current = requested
        headers = {
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "text/plain;q=0.8,application/json;q=0.7,*/*;q=0.5"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "max-age=0",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Upgrade-Insecure-Requests": "1",
            "User-Agent": user_agent,
        }
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            transport=self.transport,
            headers=headers,
        ) as client:
            for redirect_index in range(self.settings.web_fetch_max_redirects + 1):
                await self._validate_target(current)
                client.cookies.clear()
                try:
                    async with client.stream("GET", current) as response:
                        if response.status_code in REDIRECT_STATUSES:
                            location = response.headers.get("location", "").strip()
                            if not location:
                                raise PageFetchError("The page returned an empty redirect.")
                            if redirect_index >= self.settings.web_fetch_max_redirects:
                                raise PageFetchError("The page exceeded the redirect limit.")
                            current = normalize_public_url(urljoin(current, location))
                            continue
                        if response.status_code >= 400:
                            fallback_eligible = (
                                response.status_code in {401, 403, 429, 451}
                                or response.status_code >= 500
                            )
                            raise PageFetchError(
                                f"The page returned HTTP {response.status_code}.",
                                status_code=response.status_code,
                                attempt_count=attempt_count,
                                reason="http_error",
                                search_fallback_eligible=fallback_eligible,
                            )
                        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                        if content_type and content_type not in ALLOWED_CONTENT_TYPES:
                            raise PageFetchError(
                                f"Unsupported page type {content_type}; only HTML, plain text, and JSON are accepted."
                            )
                        body, download_truncated = await self._bounded_body(response)
                        final_type = content_type or self._infer_content_type(body)
                        title, text = self._extract_text(body, final_type, current, response.encoding)
                        if not text:
                            raise PageFetchError(
                                "The page contained no readable text.",
                                reason="no_readable_text",
                                search_fallback_eligible=True,
                            )
                        compact_text = " ".join(text.lower().split())
                        if final_type in {"text/html", "application/xhtml+xml"} and (
                            len(compact_text) < 500
                            and any(
                                phrase in compact_text
                                for phrase in (
                                    "enable javascript",
                                    "javascript is required",
                                    "requires javascript",
                                    "turn on javascript",
                                )
                            )
                        ):
                            raise PageFetchError(
                                "The page requires JavaScript before it exposes readable content.",
                                reason="javascript_required",
                                search_fallback_eligible=True,
                            )
                        text_truncated = len(text) > self.settings.web_fetch_max_text_chars
                        if text_truncated:
                            text = text[: self.settings.web_fetch_max_text_chars].rstrip()
                        return FetchedPage(
                            requested_url=requested,
                            final_url=current,
                            title=title or urlsplit(current).hostname or current,
                            content_text=text,
                            content_type=final_type,
                            byte_count=len(body),
                            truncated=download_truncated or text_truncated,
                            content_sha256=sha256(body).hexdigest(),
                            retrieval_attempts=attempt_count,
                        )
                except httpx.TimeoutException as exc:
                    raise PageFetchError(
                        "The page timed out before it could be read.",
                        reason="timeout",
                        search_fallback_eligible=True,
                    ) from exc
                except httpx.HTTPError as exc:
                    raise PageFetchError(
                        f"The page could not be reached: {exc}",
                        reason="unreachable",
                        search_fallback_eligible=True,
                    ) from exc
        raise PageFetchError("The page redirect loop ended unexpectedly.")

    async def _validate_target(self, url: str) -> None:
        normalized = normalize_public_url(url)
        parsed = urlsplit(normalized)
        hostname = parsed.hostname or ""
        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            addresses = await self.resolver(
                hostname,
                parsed.port
                or (
                    HTTPS_DEFAULT_PORT
                    if parsed.scheme == "https"
                    else HTTP_DEFAULT_PORT
                ),
            )
        else:
            addresses = [str(literal)]
        if not addresses:
            raise UnsafeUrlError("The page hostname did not resolve.")
        for raw_address in addresses:
            try:
                address = ipaddress.ip_address(raw_address)
            except ValueError as exc:
                raise UnsafeUrlError("The page hostname returned an invalid address.") from exc
            if not address.is_global:
                raise UnsafeUrlError(
                    "The page resolves to a local, private, reserved, or link-local address and was blocked."
                )

    @staticmethod
    async def _resolve_addresses(hostname: str, port: int) -> list[str]:
        try:
            results = await asyncio.to_thread(
                socket.getaddrinfo,
                hostname,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise PageFetchError(
                "The page hostname could not be resolved.",
                reason="dns_failure",
                search_fallback_eligible=True,
            ) from exc
        return sorted({result[4][0] for result in results})

    async def _bounded_body(self, response: httpx.Response) -> tuple[bytes, bool]:
        remaining = self.settings.web_fetch_max_bytes
        chunks: list[bytes] = []
        truncated = False
        async for chunk in response.aiter_bytes(
            chunk_size=self.settings.web_fetch_chunk_bytes
        ):
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                truncated = True
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            if remaining <= 0:
                truncated = True
                break
        return b"".join(chunks), truncated

    @staticmethod
    def _infer_content_type(body: bytes) -> str:
        sample = body.lstrip()[:WEB_CONTENT_TYPE_SAMPLE_BYTES].lower()
        if sample.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
            return "text/html"
        if sample.startswith((b"{", b"[")):
            return "application/json"
        return "text/plain"

    @staticmethod
    def _extract_text(
        body: bytes,
        content_type: str,
        url: str,
        encoding: str | None,
    ) -> tuple[str, str]:
        decoded = body.decode(encoding or "utf-8", errors="replace")
        if content_type in {"text/html", "application/xhtml+xml"}:
            parser = ReadableHTMLParser()
            parser.feed(decoded)
            parser.close()
            return parser.title, parser.text
        text = "\n".join(
            line for line in (" ".join(raw.split()) for raw in decoded.splitlines()) if line
        )
        return urlsplit(url).hostname or url, text


class WebSourceService:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        fetcher: WebPageFetcher | None = None,
    ) -> None:
        self.settings = settings
        self.storage = storage
        self.fetcher = fetcher or WebPageFetcher(settings)

    async def collect_for_prompt(
        self,
        project_id: str,
        message: str,
        progress_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        urls = extract_prompt_urls(message, self.settings.web_fetch_max_urls)

        async def collect_one(url: str) -> tuple[dict | None, dict | None]:
            await self._emit(progress_callback, {"type": "source_fetch_start", "url": url})
            try:
                normalized = normalize_public_url(url)
                cached = self.storage.latest_web_source(project_id, normalized)
                if cached is not None and self._cache_is_fresh(cached["fetched_at"]):
                    source = cached
                    await self._emit(
                        progress_callback,
                        {"type": "source_fetch_cached", **self.public_source(source)},
                    )
                else:
                    fetched = await self.fetcher.fetch(normalized)
                    source = self.storage.add_web_source(
                        project_id,
                        requested_url=fetched.requested_url,
                        final_url=fetched.final_url,
                        title=fetched.title,
                        content_text=fetched.content_text,
                        content_type=fetched.content_type,
                        byte_count=fetched.byte_count,
                        truncated=fetched.truncated,
                        content_sha256=fetched.content_sha256,
                        retrieval_method=fetched.retrieval_method,
                        retrieval_attempts=fetched.retrieval_attempts,
                    )
                    await self._emit(
                        progress_callback,
                        {"type": "source_fetch_complete", **self.public_source(source)},
                    )
                return source, None
            except WebSourceError as exc:
                failure = {
                    "url": url,
                    "detail": str(exc),
                    "retrieval_method": "direct_http",
                }
                if isinstance(exc, PageFetchError):
                    failure["attempt_count"] = exc.attempt_count
                    failure["reason"] = exc.reason
                    failure["search_fallback_eligible"] = (
                        exc.search_fallback_eligible
                    )
                    if exc.status_code is not None:
                        failure["status_code"] = exc.status_code
                await self._emit(progress_callback, {"type": "source_fetch_error", **failure})
                return None, failure

        collected = await asyncio.gather(*(collect_one(url) for url in urls))
        sources = [source for source, _failure in collected if source is not None]
        failures = [failure for _source, failure in collected if failure is not None]
        return sources, failures

    def _cache_is_fresh(self, fetched_at: str) -> bool:
        try:
            fetched = datetime.fromisoformat(fetched_at)
            if fetched.tzinfo is None:
                fetched = fetched.replace(tzinfo=UTC)
        except (TypeError, ValueError):
            return False
        age = (datetime.now(UTC) - fetched).total_seconds()
        return 0 <= age <= self.settings.web_fetch_cache_seconds

    @staticmethod
    def public_source(source: dict) -> dict:
        return {
            key: source[key]
            for key in (
                "id", "requested_url", "final_url", "title", "content_type",
                "byte_count", "char_count", "truncated", "retrieval_method",
                "retrieval_attempts", "fetched_at",
            )
            if key in source
        }

    @staticmethod
    async def _emit(
        callback: Callable[[dict[str, Any]], Awaitable[None]] | None,
        event: dict[str, Any],
    ) -> None:
        if callback is not None:
            await callback(event)
