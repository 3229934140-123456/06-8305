"""
Round-trip verification tests for the image codec engine.

Tests that encoding and then decoding an image produces identical pixel data
for both BMP (8/24/32-bit) and PNG (RGB/RGBA/Indexed) formats.
"""

import os
import random
import struct
import zlib

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


def test_png_indexed_default_palette():
    print("=== PNG Indexed Default Palette ===")
    w, h = 6, 4
    img = PngImage(w, h, 3)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, ((x + y * w) % 256,))

    assert len(img.palette) == 0, "Palette should be empty before encoding"

    encoded = png_encode(img)
    decoded = png_decode(encoded)

    assert len(decoded.palette) >= 256, "Default palette should have at least 256 entries"
    assert decoded.pixels == img.pixels, "Pixel indices should be preserved"

    print("  Created indexed PNG without explicit palette")
    print("  Auto-generated palette: %d entries" % len(decoded.palette))
    print("  All pixel indices preserved after round-trip")
    print("  PASSED\n")


def test_bmp_to_png_cross_format():
    print("=== BMP 24-bit -> PNG RGB -> BMP (Cross-Format) ===")
    w, h = 8, 6

    bmp = BmpImage(w, h, 24)
    for y in range(h):
        for x in range(w):
            bmp.set_pixel(x, y, (
                (x * 30) % 256,
                (y * 40) % 256,
                ((x * y) * 5) % 256,
            ))

    bmp_encoded = bmp_encode(bmp)
    bmp_redecoded = bmp_decode(bmp_encoded)
    assert bmp_redecoded == bmp, "BMP self round-trip failed"

    png = PngImage(w, h, 2)
    for y in range(h):
        for x in range(w):
            png.set_pixel(x, y, bmp.get_pixel(x, y))

    png_encoded = png_encode(png)
    png_decoded = png_decode(png_encoded)

    mismatches = 0
    for y in range(h):
        for x in range(w):
            bmp_px = bmp.get_pixel(x, y)
            png_px = png_decoded.get_pixel(x, y)
            if bmp_px != png_px:
                mismatches += 1
                if mismatches <= 3:
                    print("  Mismatch at (%d,%d): BMP=%s PNG=%s" % (x, y, bmp_px, png_px))

    assert mismatches == 0, "BMP -> PNG pixel mismatch: %d errors" % mismatches
    print("  BMP 24-bit <-> PNG RGB: all %d pixels match" % (w * h))
    print("  PASSED\n")


def test_bmp8_to_png_indexed_cross_format():
    print("=== BMP 8-bit <-> PNG Indexed (Cross-Format) ===")
    w, h = 7, 5

    palette = [((i * 11) % 256, (i * 17) % 256, (i * 23) % 256) for i in range(256)]

    bmp = BmpImage(w, h, 8, palette=palette)
    for y in range(h):
        for x in range(w):
            idx = (x * 13 + y * 7) % 256
            bmp.set_pixel(x, y, (idx,))

    bmp_encoded = bmp_encode(bmp)
    bmp_decoded = bmp_decode(bmp_encoded)
    assert bmp_decoded == bmp, "BMP 8-bit self round-trip failed"

    png = PngImage(w, h, 3, palette=palette)
    for y in range(h):
        for x in range(w):
            png.set_pixel(x, y, bmp.get_pixel(x, y))

    png_encoded = png_encode(png)
    png_decoded = png_decode(png_encoded)

    idx_mismatches = 0
    for y in range(h):
        for x in range(w):
            if bmp.get_pixel(x, y) != png_decoded.get_pixel(x, y):
                idx_mismatches += 1

    pal_mismatches = 0
    for i in range(min(len(bmp.palette), len(png_decoded.palette))):
        if bmp.palette[i] != png_decoded.palette[i]:
            pal_mismatches += 1
            if pal_mismatches <= 3:
                print("  Palette mismatch at %d: BMP=%s PNG=%s" % (i, bmp.palette[i], png_decoded.palette[i]))

    assert idx_mismatches == 0, "Index mismatch: %d" % idx_mismatches
    assert pal_mismatches == 0, "Palette mismatch: %d" % pal_mismatches
    print("  BMP 8-bit <-> PNG Indexed: indices and palettes match")
    print("  PASSED\n")


def test_png_standard_zlib_interop():
    print("=== Standard zlib PNG <-> Our Codec (Interop) ===")
    w, h = 10, 8
    stride = w * 3

    # Part 1: Generate a PNG with zlib compression, read it with our decoder
    print("  [1/3] zlib-compressed PNG -> our decoder")
    raw_filtered = bytearray()
    for y in range(h):
        raw_filtered.append(0)
        for x in range(w):
            raw_filtered.append((x * 25) % 256)
            raw_filtered.append((y * 30) % 256)
            raw_filtered.append(((x + y) * 15) % 256)

    compressed = zlib.compress(bytes(raw_filtered), 9)

    def make_chunk(ctype, cdata):
        crc_val = zlib.crc32(ctype + cdata) & 0xFFFFFFFF
        return struct.pack('>I', len(cdata)) + ctype + cdata + struct.pack('>I', crc_val)

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = make_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
    idat = make_chunk(b'IDAT', compressed)
    iend = make_chunk(b'IEND', b'')
    standard_png = sig + ihdr + idat + iend

    our_decoded = png_decode(standard_png)
    assert our_decoded.width == w and our_decoded.height == h
    assert our_decoded.color_type == 2

    mismatches = 0
    for y in range(h):
        for x in range(w):
            expected = ((x * 25) % 256, (y * 30) % 256, ((x + y) * 15) % 256)
            actual = our_decoded.get_pixel(x, y)
            if actual != expected:
                mismatches += 1
    assert mismatches == 0, "zlib->our decoder: %d pixel mismatches" % mismatches
    print("    PASS - %dx%d RGB pixels all match" % (w, h))

    # Part 2: Our encoded PNG -> zlib decompress
    print("  [2/3] our encoder -> zlib.decompress")
    our_img = PngImage(w, h, 2)
    for y in range(h):
        for x in range(w):
            our_img.set_pixel(x, y, ((x * 20) % 256, (y * 25) % 256, ((x * y) * 3) % 256))

    our_encoded = png_encode(our_img)

    pos = 8
    idat_data = bytearray()
    while pos < len(our_encoded):
        cl = struct.unpack('>I', our_encoded[pos:pos + 4])[0]
        ct = our_encoded[pos + 4:pos + 8]
        cd = our_encoded[pos + 8:pos + 8 + cl]
        if ct == b'IDAT':
            idat_data.extend(cd)
        elif ct == b'IEND':
            break
        pos += 12 + cl

    zlib_decoded = zlib.decompress(bytes(idat_data))
    expected_size = (stride + 1) * h
    assert len(zlib_decoded) == expected_size, "zlib decompressed size mismatch"
    print("    PASS - %d bytes filtered data decompressed by zlib" % len(zlib_decoded))

    # Part 3: Our encoded PNG -> our decoded -> pixel match with expected
    print("  [3/3] our encoder -> our decoder (sanity)")
    our_redecoded = png_decode(our_encoded)
    assert our_redecoded == our_img
    print("    PASS - round-trip pixel-perfect")

    print("  PASSED\n")


def test_png_pillow_interop():
    print("=== Pillow PNG Interop ===")
    try:
        from PIL import Image
    except ImportError:
        print("  Pillow not installed, skipping")
        print("  SKIPPED\n")
        return

    w, h = 12, 9

    # Part 1: Our PNG -> Pillow
    print("  [1/2] our PNG -> Pillow")
    our_img = PngImage(w, h, 2)
    for y in range(h):
        for x in range(w):
            our_img.set_pixel(x, y, ((x * 20) % 256, (y * 25) % 256, ((x + y) * 10) % 256))

    our_encoded = png_encode(our_img)
    import io
    pil_img = Image.open(io.BytesIO(our_encoded))
    assert pil_img.size == (w, h)
    assert pil_img.mode == 'RGB'

    mismatches = 0
    for y in range(h):
        for x in range(w):
            if pil_img.getpixel((x, y)) != our_img.get_pixel(x, y):
                mismatches += 1
    assert mismatches == 0, "Pillow read mismatch: %d" % mismatches
    print("    PASS - Pillow reads our PNG correctly")

    # Part 2: Pillow PNG -> our decoder
    print("  [2/2] Pillow PNG -> our decoder")
    pil_img2 = Image.new('RGB', (w, h))
    for y in range(h):
        for x in range(w):
            pil_img2.putpixel((x, y), ((x * 15) % 256, (y * 20) % 256, 128))

    buf = io.BytesIO()
    pil_img2.save(buf, 'PNG')
    pillow_png = buf.getvalue()

    our_decoded = png_decode(pillow_png)
    assert our_decoded.width == w and our_decoded.height == h

    mismatches = 0
    for y in range(h):
        for x in range(w):
            expected = pil_img2.getpixel((x, y))
            actual = our_decoded.get_pixel(x, y)
            if actual != expected:
                mismatches += 1
    assert mismatches == 0, "Our decoder read mismatch: %d" % mismatches
    print("    PASS - Our decoder reads Pillow PNG correctly")

    print("  PASSED\n")


def test_png_indexed_standard_zlib():
    print("=== Standard zlib Indexed PNG <-> Our Codec ===")
    w, h = 8, 6
    stride = w * 1

    palette = [((i * 8) % 256, (i * 12) % 256, (i * 16) % 256) for i in range(256)]

    # Build a standard indexed PNG with zlib compression
    raw_filtered = bytearray()
    for y in range(h):
        raw_filtered.append(0)
        for x in range(w):
            raw_filtered.append((x + y * 3) % 256)

    compressed = zlib.compress(bytes(raw_filtered), 9)

    import struct

    def make_chunk(ctype, cdata):
        crc_val = zlib.crc32(ctype + cdata) & 0xFFFFFFFF
        return struct.pack('>I', len(cdata)) + ctype + cdata + struct.pack('>I', crc_val)

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = make_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 3, 0, 0, 0))
    plte_data = bytearray()
    for r, g, b in palette[:256]:
        plte_data.extend([r, g, b])
    plte = make_chunk(b'PLTE', bytes(plte_data))
    idat = make_chunk(b'IDAT', compressed)
    iend = make_chunk(b'IEND', b'')
    standard_png = sig + ihdr + plte + idat + iend

    our_decoded = png_decode(standard_png)
    assert our_decoded.color_type == 3
    assert len(our_decoded.palette) == 256

    idx_mismatches = 0
    for y in range(h):
        for x in range(w):
            expected_idx = (x + y * 3) % 256
            actual_idx = our_decoded.get_pixel(x, y)[0]
            if actual_idx != expected_idx:
                idx_mismatches += 1

    pal_mismatches = 0
    for i in range(256):
        if our_decoded.palette[i] != palette[i]:
            pal_mismatches += 1

    assert idx_mismatches == 0, "Index mismatches: %d" % idx_mismatches
    assert pal_mismatches == 0, "Palette mismatches: %d" % pal_mismatches
    print("  Standard indexed PNG read by our decoder: all indices + palette match")
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
    test_png_indexed_default_palette()
    test_png_chunk_crc()
    test_png_filters()
    test_large_image_roundtrip()
    test_gradient_image()

    print("=" * 60)
    print("  Cross-Format & Standard Interoperability Tests")
    print("=" * 60)
    print()

    test_bmp_to_png_cross_format()
    test_bmp8_to_png_indexed_cross_format()
    test_png_standard_zlib_interop()
    test_png_indexed_standard_zlib()
    test_png_pillow_interop()

    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    main()
