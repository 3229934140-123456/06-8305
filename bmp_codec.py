"""
BMP (Bitmap) encoder and decoder.

=== BMP Format Overview ===

A BMP file consists of:
  1. File Header (14 bytes): signature 'BM', file size, reserved, pixel data offset
  2. Info Header (40 bytes, BITMAPINFOHEADER): width, height, planes, bit count,
     compression, image size, resolution, colors used, colors important
  3. Color Table (optional): for palettized images (1/4/8-bit)
  4. Pixel Data: the actual pixel rows

=== Why 4-byte Row Alignment ===

BMP requires each scanline (row) to be padded to a multiple of 4 bytes.
The row stride = ((bits_per_pixel * width + 31) // 32) * 4

Historical reason: BMP was designed for Windows on 32-bit (and earlier 16-bit)
hardware. Memory on these systems was organized in 32-bit (4-byte) words.
Aligning each row to a 4-byte boundary means that:
  - The CPU can read/write a full row using aligned 32-bit memory accesses,
    which is significantly faster on x86 architectures.
  - When using double-buffering or hardware acceleration, aligned data allows
    the display controller to use DMA transfers more efficiently.
  - If the image width in pixels does not naturally produce a byte count that
    is a multiple of 4, the remaining bytes (1-3) are filled with zero padding.

For example, a 3-pixel-wide 24-bit image needs 3*3=9 bytes per row,
but the stride is 12 bytes (next multiple of 4). 3 bytes of padding are added.

=== Bottom-Up Row Storage ===

BMP stores pixel rows from bottom to top. The first row of pixel data in the
file corresponds to the BOTTOM row of the displayed image, and the last row
corresponds to the TOP. This is called "bottom-up" orientation.

Reason: In the early days of Windows, the video framebuffer on IBM PC
compatible hardware was mapped so that the origin (0,0) was at the
bottom-left of the screen. This convention came from mathematical coordinate
systems where Y increases upward. By storing the bottom row first, BMP data
could be sent directly to the framebuffer without row reordering, enabling
efficient blitting.

A negative height value in the header indicates top-down storage (row 0 is
the top), but this is rarely used and not all viewers support it.

=== Color Depths Supported ===

  - 8-bit (indexed): Each pixel is an index into a 256-color palette.
  - 24-bit (RGB): Each pixel is 3 bytes: Blue, Green, Red (BGR order in file).
  - 32-bit (RGBA): Each pixel is 4 bytes: Blue, Green, Red, Alpha (BGRA order).

BMP also supports 1-bit and 4-bit indexed modes, but this implementation
focuses on 8/24/32-bit as the most common variants.
"""

import struct
from typing import List, Optional, Tuple


class BmpImage:
    __slots__ = ('width', 'height', 'bit_depth', 'pixels', 'palette')

    def __init__(self, width: int, height: int, bit_depth: int,
                 pixels: Optional[List[List[Tuple[int, ...]]]] = None,
                 palette: Optional[List[Tuple[int, int, int]]] = None):
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.pixels = pixels if pixels is not None else [
            [(0,) * (4 if bit_depth == 32 else (3 if bit_depth == 24 else 1))
             for _ in range(width)]
            for _ in range(height)
        ]
        self.palette = palette if palette is not None else []

    def get_pixel(self, x: int, y: int) -> Tuple[int, ...]:
        return self.pixels[y][x]

    def set_pixel(self, x: int, y: int, color: Tuple[int, ...]):
        self.pixels[y][x] = color

    def __eq__(self, other) -> bool:
        if not isinstance(other, BmpImage):
            return False
        if self.width != other.width or self.height != other.height:
            return False
        if self.bit_depth != other.bit_depth:
            return False
        if self.bit_depth == 8:
            if self.palette != other.palette:
                return False
        return self.pixels == other.pixels


def _bmp_row_stride(width: int, bit_depth: int) -> int:
    return ((bit_depth * width + 31) // 32) * 4


def bmp_decode(data: bytes) -> BmpImage:
    if len(data) < 54:
        raise ValueError("Data too short for BMP file")

    sig = data[0:2]
    if sig != b'BM':
        raise ValueError(f"Invalid BMP signature: {sig!r}")

    pixel_offset = struct.unpack_from('<I', data, 10)[0]
    header_size = struct.unpack_from('<I', data, 14)[0]
    width = struct.unpack_from('<i', data, 18)[0]
    height_raw = struct.unpack_from('<i', data, 22)[0]
    planes = struct.unpack_from('<H', data, 26)[0]
    bit_count = struct.unpack_from('<H', data, 28)[0]
    compression = struct.unpack_from('<I', data, 30)[0]

    if compression != 0:
        raise ValueError(f"Compressed BMP not supported (compression={compression})")
    if planes != 1:
        raise ValueError(f"Unsupported planes: {planes}")

    top_down = height_raw < 0
    height = abs(height_raw)

    if bit_count not in (8, 24, 32):
        raise ValueError(f"Unsupported bit depth: {bit_count}")

    palette: List[Tuple[int, int, int]] = []
    if bit_count <= 8:
        num_colors = struct.unpack_from('<I', data, 46)[0]
        if num_colors == 0:
            num_colors = 1 << bit_count
        palette_offset = 14 + header_size
        for i in range(num_colors):
            off = palette_offset + i * 4
            b = data[off]
            g = data[off + 1]
            r = data[off + 2]
            palette.append((r, g, b))

    stride = _bmp_row_stride(width, bit_count)
    pixels: List[List[Tuple[int, ...]]] = []

    for row_idx in range(height):
        if top_down:
            file_row = row_idx
        else:
            file_row = height - 1 - row_idx

        row_offset = pixel_offset + file_row * stride
        row_pixels: List[Tuple[int, ...]] = []

        if bit_count == 32:
            for x in range(width):
                off = row_offset + x * 4
                b = data[off]
                g = data[off + 1]
                r = data[off + 2]
                a = data[off + 3]
                row_pixels.append((r, g, b, a))

        elif bit_count == 24:
            for x in range(width):
                off = row_offset + x * 3
                b = data[off]
                g = data[off + 1]
                r = data[off + 2]
                row_pixels.append((r, g, b))

        elif bit_count == 8:
            for x in range(width):
                idx = data[row_offset + x]
                row_pixels.append((idx,))

        pixels.append(row_pixels)

    return BmpImage(width, height, bit_count, pixels, palette)


def bmp_encode(img: BmpImage) -> bytes:
    width = img.width
    height = img.height
    bit_depth = img.bit_depth

    stride = _bmp_row_stride(width, bit_depth)

    if bit_depth == 8:
        num_colors = len(img.palette) if img.palette else 256
        if num_colors == 0:
            num_colors = 256
    else:
        num_colors = 0

    palette_size = num_colors * 4 if num_colors > 0 else 0
    pixel_data_size = stride * height
    header_size = 40
    file_size = 14 + header_size + palette_size + pixel_data_size
    pixel_offset = 14 + header_size + palette_size

    out = bytearray()

    out += b'BM'
    out += struct.pack('<I', file_size)
    out += struct.pack('<HH', 0, 0)
    out += struct.pack('<I', pixel_offset)

    out += struct.pack('<I', header_size)
    out += struct.pack('<i', width)
    out += struct.pack('<i', height)
    out += struct.pack('<H', 1)
    out += struct.pack('<H', bit_depth)
    out += struct.pack('<I', 0)
    out += struct.pack('<I', pixel_data_size)
    out += struct.pack('<i', 2835)
    out += struct.pack('<i', 2835)
    out += struct.pack('<I', num_colors)
    out += struct.pack('<I', 0)

    if bit_depth == 8:
        for i in range(num_colors):
            if i < len(img.palette):
                r, g, b = img.palette[i]
            else:
                r, g, b = 0, 0, 0
            out += struct.pack('BBBB', b, g, r, 0)

    for row_idx in range(height - 1, -1, -1):
        row = img.pixels[row_idx]
        row_buf = bytearray(stride)

        if bit_depth == 32:
            for x, px in enumerate(row):
                r, g, b, a = px
                off = x * 4
                row_buf[off] = b
                row_buf[off + 1] = g
                row_buf[off + 2] = r
                row_buf[off + 3] = a

        elif bit_depth == 24:
            for x, px in enumerate(row):
                r, g, b = px
                off = x * 3
                row_buf[off] = b
                row_buf[off + 1] = g
                row_buf[off + 2] = r

        elif bit_depth == 8:
            for x, px in enumerate(row):
                row_buf[x] = px[0]

        out += row_buf

    return bytes(out)


def bmp_read_file(path: str) -> BmpImage:
    with open(path, 'rb') as f:
        return bmp_decode(f.read())


def bmp_write_file(path: str, img: BmpImage):
    with open(path, 'wb') as f:
        f.write(bmp_encode(img))
