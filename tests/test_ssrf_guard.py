"""SSRF guard tests: python -m unittest discover -s tests (needs requests)."""
import os
import socket
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ssrf_guard  # noqa: E402


class _Counter(BaseHTTPRequestHandler):
    hits = 0

    def do_GET(self):  # noqa: N802
        type(self).hits += 1
        self.send_response(200)
        self.send_header('content-type', 'video/mp4')
        self.end_headers()
        self.wfile.write(b'x' * 2000)

    def log_message(self, *args):
        pass


def _serve():
    _Counter.hits = 0
    server = HTTPServer(('127.0.0.1', 0), _Counter)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class GuardTests(unittest.TestCase):
    def test_refuses_private_targets_and_schemes(self):
        for url in ('http://127.0.0.1/', 'http://localhost/', 'http://[::ffff:169.254.169.254]/',
                    'http://[::1]/', 'file:///etc/passwd', 'gopher://x/', '/app/data/x.png'):
            with self.assertRaises(ValueError, msg=url):
                ssrf_guard.safe_get(url, timeout=1)

    def test_connection_is_pinned_to_the_checked_address(self):
        # DNS rebinding: public answer for the check, loopback for the connect.
        server = _serve()
        port = server.server_address[1]
        real = socket.getaddrinfo
        answers = iter(['93.184.216.34'] + ['127.0.0.1'] * 10)

        def fake(host, *args, **kwargs):
            if host == 'rebind.test':
                ip = next(answers)
                return [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, port))]
            return real(host, *args, **kwargs)

        try:
            with mock.patch('socket.getaddrinfo', side_effect=fake):
                with self.assertRaises(Exception):
                    ssrf_guard.safe_get(f'http://rebind.test:{port}/', timeout=1)
            self.assertEqual(_Counter.hits, 0, 'the second (loopback) answer must never be dialled')
        finally:
            server.shutdown()

    def test_pinned_adapter_dials_the_pinned_ip_and_keeps_the_host_header(self):
        server = _serve()
        try:
            import requests
            s = requests.Session()
            s.trust_env = False
            s.mount('http://', ssrf_guard._pinned_adapter('127.0.0.1'))
            r = s.get(f'http://media.example.invalid:{server.server_address[1]}/a.mp4', timeout=2)
            self.assertEqual(r.status_code, 200)
            self.assertEqual(_Counter.hits, 1)
        finally:
            server.shutdown()

    def test_pinned_https_keeps_the_hostname_for_sni_and_cert_check(self):
        # Prod 2026-10-28: SNI went out as the bare IP and the CDN (Cloudflare)
        # refused the handshake, so every https media download failed.
        import urllib3.connection
        adapter = ssrf_guard._pinned_adapter('203.0.113.7')
        pool = adapter.poolmanager.connection_from_url('https://cdn.example.test/a.mp4')
        conn = pool.ConnectionCls('cdn.example.test', 443)
        with mock.patch('urllib3.util.connection.create_connection', return_value=mock.MagicMock()) as dial, \
                mock.patch.object(urllib3.connection, '_ssl_wrap_socket_and_match_hostname') as wrap:
            wrap.return_value = mock.MagicMock(socket=mock.MagicMock(), is_verified=True, proxy_is_verified=None)
            conn.connect()
        self.assertEqual(dial.call_args[0][0], ('203.0.113.7', 443))
        self.assertEqual(wrap.call_args.kwargs['server_hostname'], 'cdn.example.test')
        self.assertEqual(conn.host, 'cdn.example.test')

    def test_playlists_and_reference_files_are_detected(self):
        cases = {
            b'#EXTM3U\n#EXT-X-VERSION:3\nhttp://127.0.0.1/seg.ts\n': True,
            b'\xef\xbb\xbf#EXTM3U\n': True,
            b'ffconcat version 1.0\nfile http://10.0.0.1/a.mp4\n': True,
            b"file 'http://169.254.169.254/x'\n": True,
            b'v=0\no=- 0 0 IN IP4 127.0.0.1\n': True,
            b'<?xml version="1.0"?><MPD></MPD>': True,
            b'[playlist]\nFile1=http://127.0.0.1/\n': True,
            b'\x00\x00\x00\x18ftypmp42' + b'\x00' * 100: False,
            b'ID3\x04\x00' + b'\x00' * 100: False,
            b'\x89PNG\r\n\x1a\n' + b'\x00' * 100: False,
        }
        for content, expected in cases.items():
            with tempfile.NamedTemporaryFile(delete=False) as f:
                f.write(content)
            try:
                self.assertEqual(ssrf_guard.is_reference_container(f.name), expected, content[:20])
            finally:
                os.unlink(f.name)

    def test_playlist_extensions_are_not_allowed(self):
        for ext in ('.m3u8', '.m3u', '.txt', '.ffconcat', '.sdp', '.mpd', '.pls'):
            self.assertNotIn(ext, ssrf_guard.SAFE_MEDIA_EXTENSIONS)


if __name__ == '__main__':
    unittest.main()
