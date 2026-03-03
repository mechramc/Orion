#!/usr/bin/env python3
"""T026: Verify BLOBFILE writer + weight conversion.

Tests:
  1. Header format: verify magic bytes, sizes, offsets
  2. Round-trip: float32 → blob → read back → compare fp16 fidelity
  3. Transposed conversion: verify shape handling
"""

import os
import struct
import sys
import tempfile
import unittest

import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model', 'convert'))

from hf_to_blobs_gpt2 import make_blob_header, convert_tensor_to_blob, convert_tensor_transposed


class TestBlobHeader(unittest.TestCase):
    """Test BLOBFILE header format (T024)."""

    def test_header_size(self):
        header = make_blob_header(1024)
        self.assertEqual(len(header), 128)

    def test_magic_bytes(self):
        header = make_blob_header(1024)
        self.assertEqual(header[0], 0x01)
        self.assertEqual(header[4], 0x02)
        self.assertEqual(header[64:68], bytes([0xEF, 0xBE, 0xAD, 0xDE]))
        self.assertEqual(header[68], 0x01)

    def test_data_size(self):
        header = make_blob_header(4096)
        data_size = struct.unpack_from('<I', header, 72)[0]
        self.assertEqual(data_size, 4096)

    def test_data_offset(self):
        header = make_blob_header(1024)
        offset = struct.unpack_from('<I', header, 80)[0]
        self.assertEqual(offset, 128)


class TestBlobConvert(unittest.TestCase):
    """Test blob conversion round-trip (T024)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_roundtrip_small(self):
        """Write and read back a small tensor."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, -1.5], dtype=np.float32)
        path = os.path.join(self.tmpdir, "test.bin")
        convert_tensor_to_blob(arr, path)

        # Read back
        with open(path, 'rb') as f:
            header = f.read(128)
            data = f.read()

        self.assertEqual(len(header), 128)
        result = np.frombuffer(data, dtype=np.float16)
        np.testing.assert_allclose(result.astype(np.float32), arr, atol=1e-3)

    def test_roundtrip_large(self):
        """Test with a large tensor (768x768 like GPT-2 weight)."""
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((768, 768)).astype(np.float32) * 0.02
        path = os.path.join(self.tmpdir, "large.bin")
        convert_tensor_to_blob(arr, path)

        with open(path, 'rb') as f:
            f.seek(128)
            data = f.read()

        result = np.frombuffer(data, dtype=np.float16).reshape(768, 768)
        max_err = np.max(np.abs(result.astype(np.float32) - arr))
        self.assertLess(max_err, 0.01, f"Max fp16 error: {max_err}")

    def test_file_size(self):
        """Verify file size = 128 header + N*2 fp16."""
        arr = np.zeros(1000, dtype=np.float32)
        path = os.path.join(self.tmpdir, "sized.bin")
        convert_tensor_to_blob(arr, path)
        self.assertEqual(os.path.getsize(path), 128 + 1000 * 2)


class TestTransposedConvert(unittest.TestCase):
    """Test transposed conversion for ANE conv weights."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_transpose(self):
        """Verify weight is transposed in the blob."""
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)  # [[0,1,2],[3,4,5]]
        path = os.path.join(self.tmpdir, "trans.bin")
        convert_tensor_transposed(arr, path)

        with open(path, 'rb') as f:
            f.seek(128)
            data = f.read()

        result = np.frombuffer(data, dtype=np.float16).reshape(3, 2)
        expected = arr.T  # [[0,3],[1,4],[2,5]]
        np.testing.assert_allclose(result.astype(np.float32), expected, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
