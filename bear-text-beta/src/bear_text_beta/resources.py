"""Explicit policy for external status references; no resource is fetched."""
import re
from urllib.parse import urlsplit


def is_status_badge(url):
    try:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or parsed.username or parsed.password:
            return False
        host, path = parsed.hostname, parsed.path
        return bool(
            (host == "github.com" and re.fullmatch(r"/[^/]+/[^/]+/actions/workflows/[^/]+/badge\.svg", path))
            or (host == "badge.fury.io" and path.startswith("/js/"))
            or (host == "img.shields.io" and path.startswith("/npm/"))
            or (host == "api.securityscorecards.dev" and re.fullmatch(r"/projects/github.com/[^/]+/[^/]+/badge", path))
        )
    except ValueError:
        return False


def optional_badge(block):
    return (block.type == "IMAGE" and block.metadata.get("resource_policy") == "status-badge-reference-v1"
            and is_status_badge(block.metadata.get("resource_url", "")))
