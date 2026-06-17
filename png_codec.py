"""
PNG (Portable Network Graphics) encoder and decoder.

=== PNG Chunk Structure ===

A PNG file is composed of a signature (8 bytes) followed by a sequence of
chunks. Each chunk has the following layout:

  [4 bytes] Length   - number of bytes in the chunk's data field (NOT including
                       the length/type/CRC fields themselves)
  [4 bytes] Type     - four ASCII characters identifying the chunk type
                       (e.g. "IHDR", "IDAT", "IEND", "PLTE")
  [N bytes] Data     - the chunk's payload, Length bytes long
  [4 bytes] CRC      - CRC-32 checksum computed over Type + Data

The CRC-32 is a cyclic redundancy check that detects accidental corruption.
It is computed over the chunk type and data bytes (but NOT the length field).
This allows a decoder to verify the integrity of each chunk independently.

Critical chunks (must be understood by any decoder):
  - IHDR: Image header (width, height, bit depth, color type, etc.)
  - PLTE: Palette (for indexed-color images, up to 256 entries)
  - IDAT: Image data (deflate-compressed, filtered pixel data)
  - IEND: Image end marker (empty data)

Ancillary chunks (optional, can be safely ignored):
  - tEXt, zTXt, iTXt: text metadata
  - tRNS: transparency
  - etc.

=== PNG Color Types ===

  - 0: Grayscale        - 1 channel per pixel
  - 2: Truecolor (RGB)  - 3 channels per pixel (R, G, B)
  - 3: Indexed (Palette)- 1 channel per pixel (index into PLTE)
  - 4: Grayscale+Alpha  - 2 channels per pixel
  - 6: Truecolor+Alpha  - 4 channels per pixel (R, G, B, A)

This implementation supports color types 2 (RGB), 3 (indexed), and 6 (RGBA).

=== Palette vs Truecolor ===

In truecolor mode (type 2/6), each pixel stores its color components directly.
For a 24-bit RGB image, the raw (unfiltered) data stream consists of
  R0,G0,B0, R1,G1,B1, R2,G2,B2, ...
with one filter byte prepended to each row.

In indexed/palette mode (type 3), each pixel is a single byte that serves as
an index into the PLTE chunk. The PLTE chunk contains up to 256 RGB triplets.
The raw data stream for a palettized image stores only the index bytes per
pixel (plus the per-row filter byte). To reconstruct the actual color, the
decoder must look up each index in the palette.

This separation means:
  - Palettized images are much smaller for images with ≤256 distinct colors,
    since only 1 byte per pixel is stored instead of 3 (or 4 with alpha).
  - Truecolor images can represent any of 16.7 million colors per pixel.
  - The filtering and compression stages operate on the raw sample values
    (indices for palette, color components for truecolor), which means
    filtering works the same way regardless of color type — it just operates
    on different numbers of bytes per pixel (bpp).

=== Row Filtering (Pre-compression Prediction) ===

Before compressing the pixel data with deflate, PNG applies a per-row filter
that transforms the raw pixel values into predicted residuals (differences).
The goal is to make the data more compressible: natural images tend to have
smooth gradients, so neighboring pixels are often similar. By replacing each
pixel with the difference between its actual value and a predicted value,
the residuals cluster near zero, which deflate can compress much better.

Each row in the filtered data stream starts with a filter-type byte (0–4)
followed by the filtered pixel bytes for that row. Different rows may use
different filter types. The encoder chooses the filter that minimizes the
sum of absolute values of the residuals (a simple heuristic).

The five filter types:

  Filter 0 (None):   Filt(x) = Orig(x)
    No filtering. The raw byte is stored as-is.
    Useful for random/noisy data where prediction doesn't help.

  Filter 1 (Sub):    Filt(x) = Orig(x) - Orig(x - bpp)
    Predicts the current byte from the byte 'bpp' positions to the left
    (i.e., the same channel of the previous pixel). 'bpp' = bytes per
    complete pixel (e.g., 3 for RGB, 4 for RGBA, 1 for palette).
    Works well on images with horizontal gradients.

  Filter 2 (Up):     Filt(x) = Orig(x) - Prior(x)
    Predicts the current byte from the byte directly above in the previous
    row. Works well on images with vertical gradients.

  Filter 3 (Average): Filt(x) = Orig(x) - floor((Orig(x-bpp) + Prior(x)) / 2)
    Predicts from the average of the left and above bytes. Balances
    horizontal and vertical prediction.

  Filter 4 (Paeth):  Filt(x) = Orig(x) - PaethPredictor(Orig(x-bpp), Prior(x), Prior(x-bpp))
    Uses the Paeth predictor (see below), which typically gives the best
    prediction for natural images.

In all formulas:
  - x is the byte index within the current (unfiltered) row
  - Orig(x) is the raw byte value at position x
  - Prior(x) is the raw byte value at position x in the previous row
  - bpp is bytes per complete pixel
  - If a reference position is out of bounds (before the start of the row,
    or above the first row), its value is taken as 0.

All arithmetic is modulo 256 (byte values wrap around). When decoding, the
inverse is applied: Orig(x) = (Filt(x) + Predicted(x)) mod 256.

=== Paeth Predictor ===

The Paeth predictor is a simple but effective linear predictor that chooses
from three neighboring pixels (left, above, upper-left) the one that is
closest to the linear interpolation of the three.

Given three neighbor values:
  a = left neighbor     (the byte bpp positions to the left in the same row)
  b = above neighbor    (the byte directly above in the previous row)
  c = upper-left neighbor (the byte bpp positions to the left in previous row)

The Paeth predictor computes:

  p  = a + b - c
  pa = |p - a|
  pb = |p - b|
  pc = |p - c|

  if pa <= pb and pa <= pc:  predict = a
  elif pb <= pc:             predict = b
  else:                      predict = c

Intuition: p = a + b - c is the value at the current position if the image
had a constant gradient from upper-left to lower-right. The predictor then
picks the neighbor closest to this ideal value, which effectively chooses
the direction (horizontal, vertical, or diagonal) where the gradient is
smoothest. This works remarkably well for natural images because edges tend
to be locally coherent in at least one direction.

=== Decoding: Reversing the Filter ===

Decoding reverses the filtering process row by row:

  1. Decompress the IDAT data using inflate → get filtered scanlines
  2. For each row:
     a. Read the filter type byte (0-4)
     b. For each byte position x in the row, compute:
        Orig(x) = (Filt(x) + Predicted(x)) mod 256
        where Predicted(x) depends on the filter type and the already-
        reconstructed values of the current row and the previous row.
     c. Store the reconstructed row for use by the next row's prediction.

  This is a causal process: each pixel's reconstruction depends only on
  already-reconstructed neighbors (left and above), so it can be done in
  a single pass, top to bottom, left to right.

=== IDAT and Deflate ===

The filtered pixel data is compressed with the deflate algorithm (the same
used by gzip/zlib) and stored in one or more IDAT chunks. All IDAT chunks
together form a single zlib stream — they must be concatenated before
decompression. The zlib wrapper consists of:
  - 2-byte header (CMF + FLG)
  - Deflate-compressed data
  - 4-byte Adler-32 checksum
"""

import struct
import zlib
from typing import List, Optional, Tuple

from deflate import zlib_compress, zlib_decompress

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'

FILTER_NONE = 0
FILTER_SUB = 1
FILTER_UP = 2
FILTER_AVERAGE = 3
FILTER_PAETH = 4


class PngImage:
    __slots__ = ('width', 'height', 'color_type', 'pixels', 'palette')

    def __init__(self, width: int, height: int, color_type: int,
                 pixels: Optional[List[List[Tuple[int, ...]]]] = None,
                 palette: Optional[List[Tuple[int, int, int]]] = None):
        self.width = width
        self.height = height
        self.color_type = color_type
        self.pixels = pixels if pixels is not None else self._default_pixels()
        self.palette = palette if palette is not None else []

    def _default_pixels(self) -> List[List[Tuple[int, ...]]]:
        if self.color_type == 2:
            px = (0, 0, 0)
        elif self.color_type == 6:
            px = (0, 0, 0, 255)
        elif self.color_type == 3:
            px = (0,)
        else:
            px = (0,)
        return [[px for _ in range(self.width)] for _ in range(self.height)]

    def get_pixel(self, x: int, y: int) -> Tuple[int, ...]:
        return self.pixels[y][x]

    def set_pixel(self, x: int, y: int, color: Tuple[int, ...]):
        self.pixels[y][x] = color

    def __eq__(self, other) -> bool:
        if not isinstance(other, PngImage):
            return False
        if self.width != other.width or self.height != other.height:
            return False
        if self.color_type != other.color_type:
            return False
        if self.color_type == 3:
            if self.palette != other.palette:
                return False
        return self.pixels == other.pixels


def _crc32(data: bytes) -> int:
    return zlib.crc32(data) & 0xFFFFFFFF


def _make_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    crc = _crc32(chunk_type + chunk_data)
    return (struct.pack('>I', len(chunk_data))
            + chunk_type
            + chunk_data
            + struct.pack('>I', crc))


def _bytes_per_pixel(color_type: int) -> int:
    if color_type == 2:
        return 3
    elif color_type == 6:
        return 4
    elif color_type == 3:
        return 1
    else:
        raise ValueError(f"Unsupported color type: {color_type}")


def _paeth_predictor(a: int, b: int, c: int) -> int:
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


def _filter_row(filter_type: int, row: bytes, prev_row: bytes, bpp: int) -> bytes:
    out = bytearray(len(row))
    for i in range(len(row)):
        x = row[i]
        a = row[i - bpp] if i >= bpp else 0
        b = prev_row[i] if prev_row else 0
        c = (prev_row[i - bpp] if prev_row and i >= bpp else 0)

        if filter_type == FILTER_NONE:
            out[i] = x
        elif filter_type == FILTER_SUB:
            out[i] = (x - a) & 0xFF
        elif filter_type == FILTER_UP:
            out[i] = (x - b) & 0xFF
        elif filter_type == FILTER_AVERAGE:
            out[i] = (x - ((a + b) >> 1)) & 0xFF
        elif filter_type == FILTER_PAETH:
            out[i] = (x - _paeth_predictor(a, b, c)) & 0xFF
        else:
            raise ValueError(f"Invalid filter type: {filter_type}")
    return bytes(out)


def _unfilter_row(filter_type: int, filtered: bytes, prev_row: bytes, bpp: int) -> bytes:
    out = bytearray(len(filtered))
    for i in range(len(filtered)):
        fx = filtered[i]
        a = out[i - bpp] if i >= bpp else 0
        b = prev_row[i] if prev_row else 0
        c = (prev_row[i - bpp] if prev_row and i >= bpp else 0)

        if filter_type == FILTER_NONE:
            out[i] = fx
        elif filter_type == FILTER_SUB:
            out[i] = (fx + a) & 0xFF
        elif filter_type == FILTER_UP:
            out[i] = (fx + b) & 0xFF
        elif filter_type == FILTER_AVERAGE:
            out[i] = (fx + ((a + b) >> 1)) & 0xFF
        elif filter_type == FILTER_PAETH:
            out[i] = (fx + _paeth_predictor(a, b, c)) & 0xFF
        else:
            raise ValueError(f"Invalid filter type: {filter_type}")
    return bytes(out)


def _choose_filter(row: bytes, prev_row: bytes, bpp: int) -> int:
    best = FILTER_NONE
    best_sum = float('inf')
    for ft in range(5):
        filtered = _filter_row(ft, row, prev_row, bpp)
        s = sum(filtered)
        if s < best_sum:
            best_sum = s
            best = ft
    return best


def _pixels_to_raw_rows(img: PngImage) -> List[bytes]:
    bpp = _bytes_per_pixel(img.color_type)
    rows: List[bytes] = []
    for y in range(img.height):
        row = bytearray()
        for x in range(img.width):
            px = img.pixels[y][x]
            if img.color_type == 2:
                row.extend([px[0], px[1], px[2]])
            elif img.color_type == 6:
                row.extend([px[0], px[1], px[2], px[3]])
            elif img.color_type == 3:
                row.append(px[0])
        rows.append(bytes(row))
    return rows


def _raw_rows_to_pixels(rows: List[bytes], width: int, height: int,
                        color_type: int) -> List[List[Tuple[int, ...]]]:
    bpp = _bytes_per_pixel(color_type)
    pixels: List[List[Tuple[int, ...]]] = []
    for y in range(height):
        row = rows[y]
        row_px: List[Tuple[int, ...]] = []
        for x in range(width):
            off = x * bpp
            if color_type == 2:
                row_px.append((row[off], row[off + 1], row[off + 2]))
            elif color_type == 6:
                row_px.append((row[off], row[off + 1], row[off + 2], row[off + 3]))
            elif color_type == 3:
                row_px.append((row[off],))
        pixels.append(row_px)
    return pixels


def _build_default_palette(num_colors: int) -> List[Tuple[int, int, int]]:
    palette: List[Tuple[int, int, int]] = []
    n = max(num_colors, 256)
    for i in range(n):
        r = (i * 7) % 256
        g = (i * 13) % 256
        b = (i * 23) % 256
        palette.append((r, g, b))
    return palette[:num_colors]


def _ensure_palette(img: PngImage) -> PngImage:
    if img.color_type != 3:
        return img
    if img.palette and len(img.palette) > 0:
        return img

    max_idx = 0
    for row in img.pixels:
        for px in row:
            if px[0] > max_idx:
                max_idx = px[0]
    num_colors = max(max_idx + 1, 256)
    img.palette = _build_default_palette(num_colors)
    return img


def png_encode(img: PngImage) -> bytes:
    if img.color_type == 3:
        _ensure_palette(img)

    bpp = _bytes_per_pixel(img.color_type)

    raw_rows = _pixels_to_raw_rows(img)

    filtered = bytearray()
    prev_row = b''
    for row in raw_rows:
        ft = _choose_filter(row, prev_row, bpp)
        filt_row = _filter_row(ft, row, prev_row, bpp)
        filtered.append(ft)
        filtered.extend(filt_row)
        prev_row = row

    compressed = zlib_compress(bytes(filtered))

    out = bytearray(PNG_SIGNATURE)

    ihdr_data = struct.pack('>IIBBBBB', img.width, img.height, 8,
                            img.color_type, 0, 0, 0)
    out.extend(_make_chunk(b'IHDR', ihdr_data))

    if img.color_type == 3 and img.palette:
        plte_data = bytearray()
        for r, g, b in img.palette:
            plte_data.extend([r, g, b])
        out.extend(_make_chunk(b'PLTE', bytes(plte_data)))

    out.extend(_make_chunk(b'IDAT', compressed))

    out.extend(_make_chunk(b'IEND', b''))

    return bytes(out)


def png_decode(data: bytes) -> PngImage:
    if data[:8] != PNG_SIGNATURE:
        raise ValueError("Invalid PNG signature")

    pos = 8
    chunks: List[Tuple[bytes, bytes]] = []

    while pos < len(data):
        if pos + 8 > len(data):
            raise ValueError("Truncated chunk header")
        chunk_len = struct.unpack('>I', data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        if pos + 12 + chunk_len > len(data):
            raise ValueError(f"Truncated chunk data for {chunk_type!r}")
        chunk_data = data[pos + 8:pos + 8 + chunk_len]
        stored_crc = struct.unpack('>I', data[pos + 8 + chunk_len:pos + 12 + chunk_len])[0]
        computed_crc = _crc32(chunk_type + chunk_data)
        if stored_crc != computed_crc:
            raise ValueError(
                f"CRC mismatch in chunk {chunk_type!r}: "
                f"stored=0x{stored_crc:08X}, computed=0x{computed_crc:08X}"
            )
        chunks.append((chunk_type, chunk_data))
        pos += 12 + chunk_len

    ihdr_data = None
    plte_data = None
    idat_data = bytearray()

    for ctype, cdata in chunks:
        if ctype == b'IHDR':
            ihdr_data = cdata
        elif ctype == b'PLTE':
            plte_data = cdata
        elif ctype == b'IDAT':
            idat_data.extend(cdata)
        elif ctype == b'IEND':
            break

    if ihdr_data is None:
        raise ValueError("Missing IHDR chunk")

    width = struct.unpack('>I', ihdr_data[0:4])[0]
    height = struct.unpack('>I', ihdr_data[4:8])[0]
    bit_depth = ihdr_data[8]
    color_type = ihdr_data[9]
    compression = ihdr_data[10]
    filter_method = ihdr_data[11]
    interlace = ihdr_data[12]

    if bit_depth != 8:
        raise ValueError(f"Only 8-bit depth supported, got {bit_depth}")
    if compression != 0:
        raise ValueError("Unsupported compression method")
    if filter_method != 0:
        raise ValueError("Unsupported filter method")
    if interlace != 0:
        raise ValueError("Interlacing not supported")

    bpp = _bytes_per_pixel(color_type)

    decompressed = zlib.decompress(bytes(idat_data))

    stride = width * bpp
    raw_rows: List[bytes] = []
    prev_row = b''
    offset = 0

    for y in range(height):
        if offset >= len(decompressed):
            raise ValueError("Unexpected end of filtered data")
        ft = decompressed[offset]
        offset += 1
        filtered_row = decompressed[offset:offset + stride]
        offset += stride
        raw_row = _unfilter_row(ft, filtered_row, prev_row, bpp)
        raw_rows.append(raw_row)
        prev_row = raw_row

    palette: List[Tuple[int, int, int]] = []
    if color_type == 3:
        if plte_data is None:
            raise ValueError("Indexed color type but no PLTE chunk")
        for i in range(0, len(plte_data), 3):
            palette.append((plte_data[i], plte_data[i + 1], plte_data[i + 2]))

    pixels = _raw_rows_to_pixels(raw_rows, width, height, color_type)

    return PngImage(width, height, color_type, pixels, palette)


def png_read_file(path: str) -> PngImage:
    with open(path, 'rb') as f:
        return png_decode(f.read())


def png_write_file(path: str, img: PngImage):
    with open(path, 'wb') as f:
        f.write(png_encode(img))
