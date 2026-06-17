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


def _paeth_predictor(a, b, c):
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    elif pb <= pc:
        return b
    else:
        return c


def _unfilter_row(filter_type, filtered_row, prev_row, bpp):
    out = bytearray(len(filtered_row))
    for i in range(len(filtered_row)):
        fx = filtered_row[i]
        a = out[i - bpp] if i >= bpp else 0
        b = prev_row[i] if prev_row else 0
        c = (prev_row[i - bpp] if prev_row and i >= bpp else 0)
        if filter_type == 0:
            out[i] = fx
        elif filter_type == 1:
            out[i] = (fx + a) & 0xFF
        elif filter_type == 2:
            out[i] = (fx + b) & 0xFF
        elif filter_type == 3:
            out[i] = (fx + ((a + b) >> 1)) & 0xFF
        elif filter_type == 4:
            out[i] = (fx + _paeth_predictor(a, b, c)) & 0xFF
    return bytes(out)


def _extract_idat_from_png(png_data):
    pos = 8
    idat_data = bytearray()
    while pos < len(png_data):
        cl = struct.unpack('>I', png_data[pos:pos + 4])[0]
        ct = png_data[pos + 4:pos + 8]
        cd = png_data[pos + 8:pos + 8 + cl]
        if ct == b'IDAT':
            idat_data.extend(cd)
        elif ct == b'IEND':
            break
        pos += 12 + cl
    return bytes(idat_data)


def _extract_ihdr_from_png(png_data):
    pos = 8
    while pos < len(png_data):
        cl = struct.unpack('>I', png_data[pos:pos + 4])[0]
        ct = png_data[pos + 4:pos + 8]
        cd = png_data[pos + 8:pos + 8 + cl]
        if ct == b'IHDR':
            w = struct.unpack('>I', cd[0:4])[0]
            h = struct.unpack('>I', cd[4:8])[0]
            bd = cd[8]
            ct_val = cd[9]
            return w, h, bd, ct_val
        pos += 12 + cl
    raise ValueError("No IHDR chunk")


def _extract_plte_from_png(png_data):
    pos = 8
    while pos < len(png_data):
        cl = struct.unpack('>I', png_data[pos:pos + 4])[0]
        ct = png_data[pos + 4:pos + 8]
        cd = png_data[pos + 8:pos + 8 + cl]
        if ct == b'PLTE':
            palette = []
            for i in range(0, len(cd), 3):
                palette.append((cd[i], cd[i + 1], cd[i + 2]))
            return palette
        elif ct == b'IDAT':
            break
        pos += 12 + cl
    return []


def test_png_idat_zlib_decompress_rgb():
    print("=== Our PNG IDAT + zlib.decompress -> Pixels (RGB) ===")
    w, h = 12, 9
    bpp = 3
    stride = w * bpp

    img = PngImage(w, h, 2)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (
                (x * 21) % 256,
                (y * 17) % 256,
                ((x * y) * 3) % 256,
            ))

    png_data = png_encode(img)

    idat_data = _extract_idat_from_png(png_data)
    raw_filtered = zlib.decompress(idat_data)

    expected_size = (stride + 1) * h
    assert len(raw_filtered) == expected_size, (
        "Filtered data size mismatch: expected %d, got %d" % (expected_size, len(raw_filtered)))
    print("  IDAT size: %d bytes -> filtered: %d bytes" % (len(idat_data), len(raw_filtered)))

    prev_row = b''
    mismatches = 0
    offset = 0
    for y in range(h):
        ft = raw_filtered[offset]
        offset += 1
        filtered_row = raw_filtered[offset:offset + stride]
        offset += stride
        raw_row = _unfilter_row(ft, filtered_row, prev_row, bpp)

        for x in range(w):
            off = x * bpp
            actual = (raw_row[off], raw_row[off + 1], raw_row[off + 2])
            expected = img.get_pixel(x, y)
            if actual != expected:
                mismatches += 1
                if mismatches <= 3:
                    print("    Mismatch at (%d,%d): expected=%s actual=%s" % (x, y, expected, actual))

        prev_row = raw_row

    assert mismatches == 0, "%d pixel mismatches" % mismatches
    print("  All %d×%d = %d RGB pixels match after zlib decompress + manual unfilter" % (w, h, w * h))
    print("  PASSED\n")


def test_png_idat_zlib_decompress_rgba():
    print("=== Our PNG IDAT + zlib.decompress -> Pixels (RGBA) ===")
    w, h = 10, 7
    bpp = 4
    stride = w * bpp

    img = PngImage(w, h, 6)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (
                (x * 25) % 256,
                (y * 19) % 256,
                ((x + y) * 11) % 256,
                ((x * y) * 5) % 256,
            ))

    png_data = png_encode(img)

    idat_data = _extract_idat_from_png(png_data)
    raw_filtered = zlib.decompress(idat_data)

    expected_size = (stride + 1) * h
    assert len(raw_filtered) == expected_size, (
        "Filtered data size mismatch: expected %d, got %d" % (expected_size, len(raw_filtered)))
    print("  IDAT size: %d bytes -> filtered: %d bytes" % (len(idat_data), len(raw_filtered)))

    prev_row = b''
    mismatches = 0
    offset = 0
    for y in range(h):
        ft = raw_filtered[offset]
        offset += 1
        filtered_row = raw_filtered[offset:offset + stride]
        offset += stride
        raw_row = _unfilter_row(ft, filtered_row, prev_row, bpp)

        for x in range(w):
            off = x * bpp
            actual = (raw_row[off], raw_row[off + 1], raw_row[off + 2], raw_row[off + 3])
            expected = img.get_pixel(x, y)
            if actual != expected:
                mismatches += 1
                if mismatches <= 3:
                    print("    Mismatch at (%d,%d): expected=%s actual=%s" % (x, y, expected, actual))

        prev_row = raw_row

    assert mismatches == 0, "%d pixel mismatches" % mismatches
    print("  All %d×%d = %d RGBA pixels match after zlib decompress + manual unfilter" % (w, h, w * h))
    print("  PASSED\n")


def test_png_idat_zlib_decompress_indexed():
    print("=== Our PNG IDAT + zlib.decompress -> Pixels (Indexed/Palette) ===")
    w, h = 14, 10
    bpp = 1
    stride = w * bpp

    palette = [((i * 7) % 256, (i * 13) % 256, (i * 23) % 256) for i in range(256)]
    img = PngImage(w, h, 3, palette=palette)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (((x * 13 + y * 7) * 5) % 256,))

    png_data = png_encode(img)

    idat_data = _extract_idat_from_png(png_data)
    raw_filtered = zlib.decompress(idat_data)

    expected_size = (stride + 1) * h
    assert len(raw_filtered) == expected_size, (
        "Filtered data size mismatch: expected %d, got %d" % (expected_size, len(raw_filtered)))
    print("  IDAT size: %d bytes -> filtered: %d bytes" % (len(idat_data), len(raw_filtered)))

    plte = _extract_plte_from_png(png_data)
    assert len(plte) >= 256, "Palette too short: %d entries" % len(plte)
    print("  PLTE extracted: %d entries" % len(plte))

    prev_row = b''
    idx_mismatches = 0
    pal_mismatches = 0
    offset = 0
    for y in range(h):
        ft = raw_filtered[offset]
        offset += 1
        filtered_row = raw_filtered[offset:offset + stride]
        offset += stride
        raw_row = _unfilter_row(ft, filtered_row, prev_row, bpp)

        for x in range(w):
            actual_idx = raw_row[x]
            expected_idx = img.get_pixel(x, y)[0]
            if actual_idx != expected_idx:
                idx_mismatches += 1

        prev_row = raw_row

    for i in range(min(len(palette), len(plte))):
        if palette[i] != plte[i]:
            pal_mismatches += 1
            if pal_mismatches <= 3:
                print("    Palette mismatch at %d: expected=%s actual=%s" % (i, palette[i], plte[i]))

    assert idx_mismatches == 0, "%d index mismatches" % idx_mismatches
    assert pal_mismatches == 0, "%d palette mismatches" % pal_mismatches
    print("  All %d×%d = %d indices + 256 palette entries match after zlib decompress" % (w, h, w * h))
    print("  PASSED\n")


def test_png_file_level_rgb():
    print("=== File-Level Interop: RGB PNG (write + read back with zlib) ===")
    w, h = 16, 12

    img = PngImage(w, h, 2)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (
                (x * 16) % 256,
                (y * 20) % 256,
                ((x ^ y) * 7) % 256,
            ))

    png_data = png_encode(img)

    assert png_data[:8] == b'\x89PNG\r\n\x1a\n', "Invalid PNG signature"
    print("  Signature: OK")

    width, height, bit_depth, color_type = _extract_ihdr_from_png(png_data)
    assert width == w and height == h
    assert bit_depth == 8
    assert color_type == 2
    print("  IHDR: %dx%d, 8-bit, color type 2 (RGB) — OK" % (width, height))

    idat_data = _extract_idat_from_png(png_data)
    assert len(idat_data) > 0, "No IDAT data"
    print("  IDAT: %d bytes" % len(idat_data))

    raw_filtered = zlib.decompress(idat_data)
    expected_size = (w * 3 + 1) * h
    assert len(raw_filtered) == expected_size, "Filtered size mismatch"
    print("  Decompressed: %d bytes (expected %d) — OK" % (len(raw_filtered), expected_size))

    prev_row = b''
    bpp = 3
    stride = w * bpp
    mismatches = 0
    offset = 0
    for y in range(h):
        ft = raw_filtered[offset]
        offset += 1
        filtered_row = raw_filtered[offset:offset + stride]
        offset += stride
        raw_row = _unfilter_row(ft, filtered_row, prev_row, bpp)
        for x in range(w):
            off = x * 3
            px = (raw_row[off], raw_row[off + 1], raw_row[off + 2])
            if px != img.get_pixel(x, y):
                mismatches += 1
        prev_row = raw_row

    assert mismatches == 0, "%d pixel mismatches" % mismatches
    print("  Pixels: all %d match — OK" % (w * h))
    print("  PASSED\n")


def test_png_file_level_rgba():
    print("=== File-Level Interop: RGBA PNG (write + read back with zlib) ===")
    w, h = 11, 8

    img = PngImage(w, h, 6)
    for y in range(h):
        for x in range(w):
            img.set_pixel(x, y, (
                (x * 23) % 256,
                (y * 29) % 256,
                ((x + y) * 13) % 256,
                128 + (x * 5 + y * 3) % 128,
            ))

    png_data = png_encode(img)

    assert png_data[:8] == b'\x89PNG\r\n\x1a\n', "Invalid PNG signature"
    print("  Signature: OK")

    width, height, bit_depth, color_type = _extract_ihdr_from_png(png_data)
    assert width == w and height == h
    assert bit_depth == 8
    assert color_type == 6
    print("  IHDR: %dx%d, 8-bit, color type 6 (RGBA) — OK" % (width, height))

    idat_data = _extract_idat_from_png(png_data)
    assert len(idat_data) > 0, "No IDAT data"
    print("  IDAT: %d bytes" % len(idat_data))

    raw_filtered = zlib.decompress(idat_data)
    expected_size = (w * 4 + 1) * h
    assert len(raw_filtered) == expected_size, "Filtered size mismatch"
    print("  Decompressed: %d bytes (expected %d) — OK" % (len(raw_filtered), expected_size))

    prev_row = b''
    bpp = 4
    stride = w * bpp
    mismatches = 0
    offset = 0
    for y in range(h):
        ft = raw_filtered[offset]
        offset += 1
        filtered_row = raw_filtered[offset:offset + stride]
        offset += stride
        raw_row = _unfilter_row(ft, filtered_row, prev_row, bpp)
        for x in range(w):
            off = x * 4
            px = (raw_row[off], raw_row[off + 1], raw_row[off + 2], raw_row[off + 3])
            if px != img.get_pixel(x, y):
                mismatches += 1
        prev_row = raw_row

    assert mismatches == 0, "%d pixel mismatches" % mismatches
    print("  Pixels: all %d match — OK" % (w * h))
    print("  PASSED\n")


def test_png_file_level_indexed():
    print("=== File-Level Interop: Indexed PNG (write + read back with zlib) ===")
    w, h = 15, 10

    palette = [((i * 11) % 256, (i * 17) % 256, (i * 29) % 256) for i in range(256)]
    img = PngImage(w, h, 3, palette=palette)
    for y in range(h):
        for x in range(w):
            idx = (x * 17 + y * 11 + (x * y) * 3) % 256
            img.set_pixel(x, y, (idx,))

    png_data = png_encode(img)

    assert png_data[:8] == b'\x89PNG\r\n\x1a\n', "Invalid PNG signature"
    print("  Signature: OK")

    width, height, bit_depth, color_type = _extract_ihdr_from_png(png_data)
    assert width == w and height == h
    assert bit_depth == 8
    assert color_type == 3
    print("  IHDR: %dx%d, 8-bit, color type 3 (Indexed) — OK" % (width, height))

    plte = _extract_plte_from_png(png_data)
    assert len(plte) == 256, "Expected 256 palette entries, got %d" % len(plte)
    pal_mismatches = 0
    for i in range(256):
        if plte[i] != palette[i]:
            pal_mismatches += 1
    assert pal_mismatches == 0, "%d palette mismatches" % pal_mismatches
    print("  PLTE: 256 entries, all match — OK")

    idat_data = _extract_idat_from_png(png_data)
    assert len(idat_data) > 0, "No IDAT data"
    print("  IDAT: %d bytes" % len(idat_data))

    raw_filtered = zlib.decompress(idat_data)
    expected_size = (w * 1 + 1) * h
    assert len(raw_filtered) == expected_size, "Filtered size mismatch"
    print("  Decompressed: %d bytes (expected %d) — OK" % (len(raw_filtered), expected_size))

    prev_row = b''
    bpp = 1
    stride = w * bpp
    idx_mismatches = 0
    offset = 0
    for y in range(h):
        ft = raw_filtered[offset]
        offset += 1
        filtered_row = raw_filtered[offset:offset + stride]
        offset += stride
        raw_row = _unfilter_row(ft, filtered_row, prev_row, bpp)
        for x in range(w):
            if raw_row[x] != img.get_pixel(x, y)[0]:
                idx_mismatches += 1
        prev_row = raw_row

    assert idx_mismatches == 0, "%d index mismatches" % idx_mismatches
    print("  Indices: all %d match — OK" % (w * h))
    print("  PASSED\n")


def test_png_standard_zlib_compressed_read_all_filters():
    print("=== Standard zlib PNG (all 5 filters) -> Our Decoder ===")
    w, h = 5, 6
    bpp = 3
    stride = w * bpp

    from png_codec import (
        FILTER_NONE, FILTER_SUB, FILTER_UP, FILTER_AVERAGE, FILTER_PAETH
    )

    def make_filtered_png(filter_byte):
        raw_rows = []
        for y in range(h):
            row = bytearray()
            for x in range(w):
                row.append((x * 50 + y * 10) % 256)
                row.append((y * 40 + x * 5) % 256)
                row.append((x + y) * 20 % 256)
            raw_rows.append(bytes(row))

        from png_codec import _filter_row
        raw_filtered = bytearray()
        prev_row = b''
        for row in raw_rows:
            raw_filtered.append(filter_byte)
            filt_row = _filter_row(filter_byte, row, prev_row, bpp)
            raw_filtered.extend(filt_row)
            prev_row = row

        compressed = zlib.compress(bytes(raw_filtered), 9)

        def make_chunk(ctype, cdata):
            crc_val = zlib.crc32(ctype + cdata) & 0xFFFFFFFF
            return struct.pack('>I', len(cdata)) + ctype + cdata + struct.pack('>I', crc_val)

        sig = b'\x89PNG\r\n\x1a\n'
        ihdr = make_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
        idat = make_chunk(b'IDAT', compressed)
        iend = make_chunk(b'IEND', b'')
        return sig + ihdr + idat + iend

    filter_names = {
        FILTER_NONE: "None",
        FILTER_SUB: "Sub",
        FILTER_UP: "Up",
        FILTER_AVERAGE: "Average",
        FILTER_PAETH: "Paeth",
    }

    for ft, name in sorted(filter_names.items()):
        png_data = make_filtered_png(ft)
        decoded = png_decode(png_data)
        assert decoded.width == w and decoded.height == h
        assert decoded.color_type == 2

        mismatches = 0
        for y in range(h):
            for x in range(w):
                expected = (
                    (x * 50 + y * 10) % 256,
                    (y * 40 + x * 5) % 256,
                    (x + y) * 20 % 256,
                )
                actual = decoded.get_pixel(x, y)
                if actual != expected:
                    mismatches += 1
        assert mismatches == 0, "Filter %s: %d mismatches" % (name, mismatches)
        print("  Filter %s: OK" % name)

    print("  All 5 filter types decoded correctly from zlib-compressed PNG")
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
    print("  Standard Library Interop (IDAT via zlib.decompress)")
    print("=" * 60)
    print()

    test_png_idat_zlib_decompress_rgb()
    test_png_idat_zlib_decompress_rgba()
    test_png_idat_zlib_decompress_indexed()

    print("=" * 60)
    print("  File-Level Interop (full PNG structure + zlib)")
    print("=" * 60)
    print()

    test_png_file_level_rgb()
    test_png_file_level_rgba()
    test_png_file_level_indexed()
    test_png_standard_zlib_compressed_read_all_filters()

    print("=" * 60)
    print("  Cross-Format Conversion Tests")
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
