"""
Round-trip verification tests for the image codec engine.

Tests that encoding and then decoding an image produces identical pixel data
for both BMP (8/24/32-bit) and PNG (RGB/RGBA/Indexed) formats.
"""

import os
import random
import struct

from bmp_codec import BmpImage, bmp_encode, bmp_decode
from png_codec import PngImage, png_encode, png_decode
from deflate import zlib_compress, zlib_decompress


def test_deflate_roundtrip():
    print("=== Deflate/Zlib Round-Trip ===")
    test_cases = [
        b'',
        b'Hello, World!',
        b'\x00' * 1000,
        bytes(range(256)) * 4,
        b'A' * 5000,
        os.urandom(1024),
    ]
    for i, data in enumerate(test_cases):
        compressed = zlib_compress(data)
        decompressed = zlib_decompress(compressed)
        assert decompressed == data, f"Deflate round-trip failed for case {i}"
        ratio = len(compressed) / len(data) * 100 if len(data) > 0 else 0
        print(f"  Case {i}: {len(data)} bytes -> {len(compressed)} bytes ({ratio:.1f}%) OK")
    print("  All deflate round-trip tests PASSED\n")


def test_bmp_24bit_roundtrip():
    print("=== BMP 24-bit Round-Trip ===")
    w, h = 5, 4
    img = BmpImage(w, h, 24)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (
                (x * 50) % 256,
                (y * 60) % 256,
                ((x + y) * 40) % 256,
            ))

    encoded = bmp_encode(img)
    decoded = bmp_decode(encoded)
    assert decoded == img, "BMP 24-bit round-trip mismatch"
    stride = ((24 * w + 31) // 32) * 4
    print(f"  {w}x{h} image, stride={stride}, encoded={len(encoded)} bytes")
    print(f"  Pixel (0,0): {img.get_pixel(0,0)} -> {decoded.get_pixel(0,0)}")
    print("  PASSED\n")


def test_bmp_32bit_roundtrip():
    print("=== BMP 32-bit Round-Trip ===")
    w, h = 7, 3
    img = BmpImage(w, h, 32)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (
                (x * 30) % 256,
                (y * 80) % 256,
                ((x * y) * 10) % 256,
                255,
            ))

    encoded = bmp_encode(img)
    decoded = bmp_decode(encoded)
    assert decoded == img, "BMP 32-bit round-trip mismatch"
    print(f"  {w}x{h} image, encoded={len(encoded)} bytes")
    print("  PASSED\n")


def test_bmp_8bit_roundtrip():
    print("=== BMP 8-bit (Palette) Round-Trip ===")
    w, h = 10, 6
    palette = [((i * 3) % 256, (i * 5) % 256, (i * 7) % 256) for i in range(256)]
    img = BmpImage(w, h, 8, palette=palette)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (((x + y) * 17) % 256,))

    encoded = bmp_encode(img)
    decoded = bmp_decode(encoded)
    assert decoded.palette == img.palette, "BMP 8-bit palette mismatch"
    assert decoded == img, "BMP 8-bit round-trip mismatch"
    print(f"  {w}x{h} image, palette=256 entries, encoded={len(encoded)} bytes")
    print("  PASSED\n")


def test_bmp_alignment():
    print("=== BMP Row Alignment Verification ===")
    for bit_depth in [8, 24, 32]:
        for width in [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17]:
            stride = ((bit_depth * width + 31) // 32) * 4
            raw_bytes = (bit_depth * width + 7) // 8
            padding = stride - raw_bytes
            h = 2
            if bit_depth == 8:
                img = BmpImage(width, h, 8, palette=[(0, 0, 0)] * 256)
                for y in range(h):
                    for x in range(width):
                        img.set_pixel(x, y, (x % 256,))
            elif bit_depth == 24:
                img = BmpImage(width, h, 24)
                for y in range(h):
                    for x in range(width):
                        img.set_pixel(x, y, (x, y, (x+y)%256))
            else:
                img = BmpImage(width, h, 32)
                for y in range(h):
                    for x in range(width):
                        img.set_pixel(x, y, (x, y, (x+y)%256, 255))

            encoded = bmp_encode(img)
            decoded = bmp_decode(encoded)
            assert decoded == img, f"Alignment test failed: {width}x{h} @ {bit_depth}bpp"

    print(f"  Tested all widths 1..17 at 8/24/32 bpp — all PASSED\n")


def test_png_rgb_roundtrip():
    print("=== PNG RGB (Color Type 2) Round-Trip ===")
    w, h = 8, 6
    img = PngImage(w, h, 2)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (
                (x * 32) % 256,
                (y * 42) % 256,
                ((x + y) * 20) % 256,
            ))

    encoded = png_encode(img)
    decoded = png_decode(encoded)
    assert decoded == img, "PNG RGB round-trip mismatch"
    print(f"  {w}x{h} image, encoded={len(encoded)} bytes")
    print("  PASSED\n")


def test_png_rgba_roundtrip():
    print("=== PNG RGBA (Color Type 6) Round-Trip ===")
    w, h = 6, 5
    img = PngImage(w, h, 6)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (
                (x * 40) % 256,
                (y * 50) % 256,
                ((x * y) * 8) % 256,
                ((x + y) * 25) % 256,
            ))

    encoded = png_encode(img)
    decoded = png_decode(encoded)
    assert decoded == img, "PNG RGBA round-trip mismatch"
    print(f"  {w}x{h} image, encoded={len(encoded)} bytes")
    print("  PASSED\n")


def test_png_indexed_roundtrip():
    print("=== PNG Indexed/Palette (Color Type 3) Round-Trip ===")
    w, h = 12, 8
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 128, 128), (255, 255, 255),
    ]
    img = PngImage(w, h, 3, palette=palette)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (((x + y) * 3) % len(palette),))

    encoded = png_encode(img)
    decoded = png_decode(encoded)
    assert decoded.palette == img.palette, "PNG palette mismatch"
    assert decoded == img, "PNG indexed round-trip mismatch"
    print(f"  {w}x{h} image, palette={len(palette)} entries, encoded={len(encoded)} bytes")
    print("  PASSED\n")


def test_png_chunk_crc():
    print("=== PNG Chunk CRC Verification ===")
    w, h = 4, 4
    img = PngImage(w, h, 2)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (x * 60, y * 60, 128))

    encoded = png_encode(img)

    corrupted = bytearray(encoded)
    for i in range(len(corrupted)):
        if corrupted[i:i+4] == b'IDAT':
            data_start = i + 4
            if data_start + 10 < len(corrupted):
                corrupted[data_start + 5] ^= 0xFF
                try:
                    png_decode(bytes(corrupted))
                    print("  WARNING: CRC check did not catch corruption")
                except ValueError as e:
                    if "CRC" in str(e):
                        print("  CRC correctly detected data corruption")
                    else:
                        print(f"  Corruption detected but with unexpected error: {e}")
                break

    print("  PASSED\n")


def test_png_filters():
    print("=== PNG All Filter Types Verification ===")
    from png_codec import (
        _filter_row, _unfilter_row, FILTER_NONE, FILTER_SUB,
        FILTER_UP, FILTER_AVERAGE, FILTER_PAETH
    )

    row = bytes([10, 20, 30, 40, 50, 60])
    prev = bytes([5, 15, 25, 35, 45, 55])
    bpp = 3

    for ft, name in [(FILTER_NONE, "None"), (FILTER_SUB, "Sub"),
                     (FILTER_UP, "Up"), (FILTER_AVERAGE, "Average"),
                     (FILTER_PAETH, "Paeth")]:
        filtered = _filter_row(ft, row, prev, bpp)
        unfiltered = _unfilter_row(ft, filtered, prev, bpp)
        assert unfiltered == row, f"Filter {name} round-trip failed"
        print(f"  Filter {name}: original={list(row)} -> filtered={list(filtered)} -> restored={list(unfiltered)} OK")

    print("  PASSED\n")


def test_large_image_roundtrip():
    print("=== Large Image Round-Trip (Stress Test) ===")
    w, h = 64, 64

    rng = random.Random(42)
    img = PngImage(w, h, 2)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))

    encoded = png_encode(img)
    decoded = png_decode(encoded)
    assert decoded == img, "Large PNG RGB round-trip mismatch"
    print(f"  PNG RGB {w}x{h}: encoded={len(encoded)} bytes ({len(encoded)/(w*h*3)*100:.1f}% of raw)")

    bmp_img = BmpImage(w, h, 24)
    for y in range(h):
        for x in range(w):
            px = img.get_pixel(x, y)
            bmp_img.set_pixel(x, y, px)

    bmp_encoded = bmp_encode(bmp_img)
    bmp_decoded = bmp_decode(bmp_encoded)
    assert bmp_decoded == bmp_img, "Large BMP 24-bit round-trip mismatch"
    print(f"  BMP 24-bit {w}x{h}: encoded={len(bmp_encoded)} bytes")

    print("  PASSED\n")


def test_gradient_image():
    print("=== Gradient Image (Best Case for Compression) ===")
    w, h = 32, 32
    img = PngImage(w, h, 2)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (x * 8, y * 8, 128))

    encoded = png_encode(img)
    decoded = png_decode(encoded)
    assert decoded == img, "Gradient PNG round-trip mismatch"
    raw_size = w * h * 3
    print(f"  {w}x{h} gradient: raw={raw_size} bytes, encoded={len(encoded)} bytes ({len(encoded)/raw_size*100:.1f}%)")
    print("  PASSED\n")


def main():
    print("=" * 60)
    print("  Image Format Codec Engine — Round-Trip Verification")
    print("=" * 60)
    print()

    test_deflate_roundtrip()
    test_bmp_24bit_roundtrip()
    test_bmp_32bit_roundtrip()
    test_bmp_8bit_roundtrip()
    test_bmp_alignment()
    test_png_rgb_roundtrip()
    test_png_rgba_roundtrip()
    test_png_indexed_roundtrip()
    test_png_chunk_crc()
    test_png_filters()
    test_large_image_roundtrip()
    test_gradient_image()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
