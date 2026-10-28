"""Image stack + upload cap: python -m unittest discover -s tests (needs requirements.txt minus markitdown)."""
import io
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['CONVERTER_MAX_CONTENT_MB'] = '1'  # read at import: keep the 413 test small

# app first, as in prod: it imports pillow_heif before cv2 (cv2 first makes
# HEIF *encoding* segfault on macOS wheels, old and new; the app only decodes).
import app as converter  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, features  # noqa: E402

FIXED_LIBWEBP = (1, 3, 2)  # CVE-2023-4863


def _ver(text):
    return tuple(int(x) for x in re.findall(r'\d+', text)[:3])


def _png(size=(64, 48), color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, format='PNG')
    return buf.getvalue()


class LibwebpVersions(unittest.TestCase):
    def test_pillow_libwebp_is_patched(self):
        self.assertGreaterEqual(_ver(features.version('webp')), FIXED_LIBWEBP)

    def test_opencv_libwebp_is_patched(self):
        # OpenCV's build info only shows libwebp ABI numbers; 4.8.1 is the
        # release whose bundled 3rdparty libwebp moved to 1.3.2.
        self.assertGreaterEqual(_ver(cv2.__version__), (4, 8, 1))


class ImageOps(unittest.TestCase):
    def test_webp_and_heif_round_trip(self):
        buf = io.BytesIO()
        Image.new('RGB', (32, 32), (1, 2, 3)).save(buf, format='WEBP')
        self.assertEqual(Image.open(io.BytesIO(buf.getvalue())).size, (32, 32))
        heif = io.BytesIO()
        Image.new('RGB', (32, 32)).save(heif, format='HEIF')  # register_heif_opener() ran in app
        self.assertEqual(Image.open(io.BytesIO(heif.getvalue())).format, 'HEIF')

    def test_opencv_decode_encode(self):
        ok, enc = cv2.imencode('.webp', np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertTrue(ok)
        self.assertEqual(cv2.imdecode(enc, cv2.IMREAD_COLOR).shape, (8, 8, 3))


class UploadLimit(unittest.TestCase):
    def setUp(self):
        self.client = converter.app.test_client()

    def test_convert_image_still_works(self):
        r = self.client.post('/convert-image', data={
            'file': (io.BytesIO(_png()), 'a.png'), 'outputFormat': 'webp', 'width': '32', 'height': '32',
        }, content_type='multipart/form-data')
        self.assertEqual(r.status_code, 200, r.data[:200])
        self.assertEqual(Image.open(io.BytesIO(r.data)).size, (32, 32))

    def test_oversized_body_is_refused_with_413(self):
        big = b'\0' * (1024 * 1024 + 1)
        r = self.client.post('/convert-image', data={'file': (io.BytesIO(big), 'a.png')},
                             content_type='multipart/form-data')
        self.assertEqual(r.status_code, 413)
        self.assertIn('exceeds', r.get_json()['error'])


if __name__ == '__main__':
    unittest.main()
