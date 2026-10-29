"""
SSRF guard for server-side URL fetches.

The converter fetches caller-supplied media URLs (videoUrl / imageUrl / OG
background_url). An attacker (or a prompt-injected agent) could point those at
internal-only addresses (cloud metadata 169.254.169.254, the docker-network
services, the WG-peered boxes) to exfiltrate internal data. `safe_get` resolves
the host and refuses any URL that maps to a private/loopback/link-local/reserved
address, re-validating on every redirect hop.

The connection is pinned to the address that was checked (no second DNS
lookup by requests), so a host that answers with a public IP for the check and
127.0.0.1 for the connect (DNS rebinding) cannot slip through. TLS still uses
the real hostname for SNI and certificate verification.

Public URLs (DO Spaces / replicate.delivery / normal CDNs) resolve to public IPs
and pass unchanged, so legitimate fetches are NOT blocked.
"""
import ipaddress
import socket
from urllib.parse import urlparse, urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError
from urllib3.util import connection as urllib3_connection_util
from socket import timeout as SocketTimeout


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
    to public addresses. Returns the URL; see `resolve_public` for the address."""
    resolve_public(url)
    return url


def resolve_public(url):
    """Validate `url` like assert_public_url and return one checked address to
    connect to."""
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
    return sorted(addrs)[0]


def _dial(conn, ip):
    """Open the TCP socket to `ip` for urllib3 connection `conn`. `conn._dns_host`
    is left alone: urllib3's `host` property (Host header, TLS SNI, certificate
    hostname check) reads it, so overwriting it sent the bare IP as SNI and
    Cloudflare-fronted CDNs refused the handshake (prod 2026-10-28)."""
    try:
        sock = urllib3_connection_util.create_connection(
            (ip, conn.port),
            conn.timeout,
            source_address=conn.source_address,
            socket_options=conn.socket_options,
        )
    except SocketTimeout as e:
        raise ConnectTimeoutError(
            conn, f"Connection to {conn.host} timed out. (connect timeout={conn.timeout})") from e
    except OSError as e:
        raise NewConnectionError(conn, f"Failed to establish a new connection: {e}") from e
    return sock


def _pinned_adapter(ip):
    """A requests adapter whose connections dial `ip` whatever the URL's host
    resolves to now, while the Host header, SNI and certificate checks keep
    using the URL's hostname."""
    class PinnedHTTPConnection(HTTPConnection):
        def _new_conn(self):
            return _dial(self, ip)

    class PinnedHTTPSConnection(HTTPSConnection):
        def _new_conn(self):
            return _dial(self, ip)

    class PinnedHTTPPool(HTTPConnectionPool):
        ConnectionCls = PinnedHTTPConnection

    class PinnedHTTPSPool(HTTPSConnectionPool):
        ConnectionCls = PinnedHTTPSConnection

    class PinnedAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            super().init_poolmanager(*args, **kwargs)
            self.poolmanager.pool_classes_by_scheme = {'http': PinnedHTTPPool, 'https': PinnedHTTPSPool}

    return PinnedAdapter()


def safe_get(url, max_redirects=5, **kwargs):
    """Drop-in replacement for requests.get that validates the target (and every
    redirect hop) is a public address before fetching. Follows redirects manually
    so each Location is re-checked. Pass-through kwargs (timeout, stream, headers)."""
    kwargs.pop("allow_redirects", None)
    current = url
    for _ in range(max_redirects + 1):
        ip = resolve_public(current)
        session = requests.Session()
        session.trust_env = False  # never via HTTP(S)_PROXY: the proxy would resolve the host itself
        adapter = _pinned_adapter(ip)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        resp = session.get(current, allow_redirects=False, **kwargs)
        if resp.status_code in (301, 302, 303, 307, 308) and "location" in resp.headers:
            nxt = urljoin(current, resp.headers["location"])
            resp.close()
            current = nxt
            continue
        return resp
    raise ValueError("SSRF_BLOCKED: too many redirects")


# Extensions a downloaded file may be saved under (ffmpeg picks demuxers by
# extension as well as content).
SAFE_MEDIA_EXTENSIONS = {
    '.mp4', '.mov', '.m4v', '.webm', '.mkv', '.avi', '.gif', '.mpeg', '.mpg', '.ts',
    '.mp3', '.wav', '.m4a', '.aac', '.ogg', '.oga', '.opus', '.flac', '.weba',
    '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff', '.heic', '.avif',
}

# Text formats ffmpeg treats as pointers to other inputs (HLS/M3U playlists,
# ffconcat lists, SDP, DASH/ISM manifests, XSPF/PLS playlists).
_REFERENCE_MARKERS = (b'#extm3u', b'#ext-x-', b'ffconcat', b'file ', b'v=0', b'<?xml', b'<mpd', b'<smil', b'[playlist]', b'<playlist')


def is_reference_container(path):
    """True when the file is a playlist / reference list rather than media."""
    try:
        with open(path, 'rb') as f:
            head = f.read(4096)
    except OSError:
        return True
    text = head.lstrip(b'\xef\xbb\xbf \t\r\n').lower()
    return any(text.startswith(m) for m in _REFERENCE_MARKERS) or b'#extm3u' in head.lower()
