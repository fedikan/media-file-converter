"""
SSRF guard for server-side URL fetches.

The converter fetches caller-supplied media URLs (videoUrl / imageUrl / OG
background_url). An attacker (or a prompt-injected agent) could point those at
internal-only addresses (cloud metadata 169.254.169.254, the docker-network
services, the WG-peered boxes) to exfiltrate internal data. `safe_get` resolves
the host and refuses any URL that maps to a private/loopback/link-local/reserved
address, re-validating on every redirect hop.

Public URLs (DO Spaces / replicate.delivery / normal CDNs) resolve to public IPs
and pass unchanged, so legitimate fetches are NOT blocked.
"""
import ipaddress
import socket
from urllib.parse import urlparse, urljoin

import requests


def _is_public_ip(ip_str):
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local      # 169.254/16 incl. 169.254.169.254 metadata
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def assert_public_url(url):
    """Raise ValueError unless `url` is an http(s) URL whose host resolves only
    to public addresses."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("SSRF_BLOCKED: only http(s) URLs are allowed")
    host = parsed.hostname
    if not host:
        raise ValueError("SSRF_BLOCKED: missing host")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"SSRF_BLOCKED: cannot resolve host: {e}")
    addrs = {info[4][0] for info in infos}
    if not addrs:
        raise ValueError("SSRF_BLOCKED: host did not resolve")
    for addr in addrs:
        if not _is_public_ip(addr):
            raise ValueError(f"SSRF_BLOCKED: {host} resolves to non-public address {addr}")
    return url


def safe_get(url, max_redirects=5, **kwargs):
    """Drop-in replacement for requests.get that validates the target (and every
    redirect hop) is a public address before fetching. Follows redirects manually
    so each Location is re-checked. Pass-through kwargs (timeout, stream, headers)."""
    kwargs.pop("allow_redirects", None)
    current = url
    for _ in range(max_redirects + 1):
        assert_public_url(current)
        resp = requests.get(current, allow_redirects=False, **kwargs)
        if resp.status_code in (301, 302, 303, 307, 308) and "location" in resp.headers:
            nxt = urljoin(current, resp.headers["location"])
            resp.close()
            current = nxt
            continue
        return resp
    raise ValueError("SSRF_BLOCKED: too many redirects")
