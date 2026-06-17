"""
Deflate/Inflate compression for PNG IDAT chunks.

PNG requires a zlib wrapper (CMF + FLG + deflate data + Adler-32 checksum)
around the raw deflate-compressed stream.

This module implements:
  - deflate: raw deflate compression (stored + fixed Huffman blocks)
  - inflate: raw deflate decompression
  - zlib_compress / zlib_decompress: zlib-wrapped versions for PNG

Huffman coding notes:
  - Fixed Huffman codes (RFC 1951 section 3.2.6) are used for simplicity.
  - Literal/length codes 0..287 map to specific bit patterns.
  - Distance codes 0..29 use 5-bit codes.
  - For lengths >= 3, a length code (257..285) plus extra bits encode the
    actual match length; similarly distance codes carry extra bits.

Bit packing in deflate:
  - Bits are packed LSB-first into bytes.
  - When writing a Huffman code, the code's bits are emitted with the
    least-significant bit first, which is why we reverse the code bits
    before writing them.
"""

import struct
from typing import List, Tuple

FIXED_LIT_LEN_TABLE: List[Tuple[int, int]] = []
FIXED_DIST_TABLE: List[Tuple[int, int]] = []

def _build_fixed_tables():
    if FIXED_LIT_LEN_TABLE:
        return
    for i in range(288):
        if i <= 143:
            FIXED_LIT_LEN_TABLE.append((8, i + 48))
        elif i <= 255:
            FIXED_LIT_LEN_TABLE.append((9, i - 144 + 400))
        elif i <= 279:
            FIXED_LIT_LEN_TABLE.append((7, i - 256 + 0))
        else:
            FIXED_LIT_LEN_TABLE.append((8, i - 280 + 192))
    for i in range(32):
        FIXED_DIST_TABLE.append((5, i))

_build_fixed_tables()

LENGTH_BASE = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31,
    35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
]
LENGTH_EXTRA = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
    3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
]
DIST_BASE = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97,
    129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073,
    4097, 6145, 8193, 12289, 16385, 24577,
]
DIST_EXTRA = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
    6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
]


def _reverse_bits(value: int, num_bits: int) -> int:
    result = 0
    for _ in range(num_bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


class _BitWriter:
    __slots__ = ('_buf', '_byte', '_bit_pos')

    def __init__(self):
        self._buf = bytearray()
        self._byte = 0
        self._bit_pos = 0

    def write_bits(self, value: int, num_bits: int):
        for i in range(num_bits):
            if value & (1 << i):
                self._byte |= (1 << self._bit_pos)
            self._bit_pos += 1
            if self._bit_pos == 8:
                self._buf.append(self._byte)
                self._byte = 0
                self._bit_pos = 0

    def write_bits_reversed(self, code: int, num_bits: int):
        self.write_bits(_reverse_bits(code, num_bits), num_bits)

    def flush(self) -> bytes:
        if self._bit_pos > 0:
            self._buf.append(self._byte)
            self._byte = 0
            self._bit_pos = 0
        return bytes(self._buf)


class _BitReader:
    __slots__ = ('_data', '_byte_pos', '_bit_pos')

    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0

    def read_bits(self, num_bits: int) -> int:
        result = 0
        for i in range(num_bits):
            if self._byte_pos >= len(self._data):
                raise ValueError("Unexpected end of deflate stream")
            bit = (self._data[self._byte_pos] >> self._bit_pos) & 1
            result |= (bit << i)
            self._bit_pos += 1
            if self._bit_pos == 8:
                self._byte_pos += 1
                self._bit_pos = 0
        return result

    def align_to_byte(self):
        if self._bit_pos > 0:
            self._byte_pos += 1
            self._bit_pos = 0

    def read_bytes(self, n: int) -> bytes:
        self.align_to_byte()
        if self._byte_pos + n > len(self._data):
            raise ValueError("Unexpected end of deflate stream")
        result = self._data[self._byte_pos:self._byte_pos + n]
        self._byte_pos += n
        return result


def _find_length_code(length: int) -> Tuple[int, int]:
    for i in range(len(LENGTH_BASE) - 1, -1, -1):
        if length >= LENGTH_BASE[i]:
            return (257 + i, length - LENGTH_BASE[i])
    raise ValueError(f"Invalid length: {length}")


def _find_dist_code(distance: int) -> Tuple[int, int]:
    for i in range(len(DIST_BASE) - 1, -1, -1):
        if distance >= DIST_BASE[i]:
            return (i, distance - DIST_BASE[i])
    raise ValueError(f"Invalid distance: {distance}")


def _lz77_compress(data: bytes, window_size: int = 32768) -> List:
    tokens: List = []
    i = 0
    n = len(data)
    hash_table = {}

    while i < n:
        best_len = 0
        best_dist = 0
        max_match = min(258, n - i)
        if max_match >= 3:
            h = ((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]) if i + 2 < n else -1
            if h in hash_table:
                for j in hash_table[h]:
                    if i - j > window_size:
                        continue
                    k = 0
                    while k < max_match and data[j + k] == data[i + k]:
                        k += 1
                    if k > best_len:
                        best_len = k
                        best_dist = i - j
            if h >= 0:
                hash_table.setdefault(h, []).append(i)
                if len(hash_table[h]) > 32:
                    hash_table[h] = hash_table[h][-32:]

        if best_len >= 3:
            tokens.append(('match', best_len, best_dist))
            for off in range(1, best_len):
                pos = i + off
                if pos + 2 < n:
                    h2 = ((data[pos] << 16) | (data[pos + 1] << 8) | data[pos + 2])
                    hash_table.setdefault(h2, []).append(pos)
                    if len(hash_table[h2]) > 32:
                        hash_table[h2] = hash_table[h2][-32:]
            i += best_len
        else:
            tokens.append(('lit', data[i]))
            i += 1
    return tokens


def deflate_compress(data: bytes) -> bytes:
    tokens = _lz77_compress(data) if data else []
    writer = _BitWriter()

    writer.write_bits(1, 1)
    writer.write_bits(1, 2)

    for token in tokens:
        if token[0] == 'lit':
            sym = token[1]
            bit_len, code = FIXED_LIT_LEN_TABLE[sym]
            writer.write_bits_reversed(code, bit_len)
        else:
            _, length, distance = token
            len_code, len_extra = _find_length_code(length)
            bit_len, code = FIXED_LIT_LEN_TABLE[len_code]
            writer.write_bits_reversed(code, bit_len)
            extra_bits_count = LENGTH_EXTRA[len_code - 257]
            if extra_bits_count > 0:
                writer.write_bits(len_extra, extra_bits_count)
            dist_code, dist_extra = _find_dist_code(distance)
            bit_len_d, code_d = FIXED_DIST_TABLE[dist_code]
            writer.write_bits_reversed(code_d, bit_len_d)
            extra_dist_bits = DIST_EXTRA[dist_code]
            if extra_dist_bits > 0:
                writer.write_bits(dist_extra, extra_dist_bits)

    bit_len_e, code_e = FIXED_LIT_LEN_TABLE[256]
    writer.write_bits_reversed(code_e, bit_len_e)

    return writer.flush()


def _decode_huffman_symbol(reader: _BitReader, table: List[Tuple[int, int]]) -> int:
    code = 0
    for bit_len in range(1, 16):
        code = (code << 1) | reader.read_bits(1)
        for sym, (bl, sym_code) in enumerate(table):
            if bl == bit_len and sym_code == code:
                return sym
    raise ValueError("Invalid Huffman code in deflate stream")


def deflate_decompress(data: bytes) -> bytes:
    reader = _BitReader(data)
    output = bytearray()

    while True:
        bfinal = reader.read_bits(1)
        btype = reader.read_bits(2)

        if btype == 0:
            reader.align_to_byte()
            raw_len = reader.read_bits(16)
            nlen = reader.read_bits(16)
            if raw_len != (nlen ^ 0xFFFF):
                raise ValueError("Invalid stored block lengths")
            for _ in range(raw_len):
                output.append(reader.read_bits(8))

        elif btype == 1:
            while True:
                sym = _decode_huffman_symbol(reader, FIXED_LIT_LEN_TABLE)
                if sym < 256:
                    output.append(sym)
                elif sym == 256:
                    break
                else:
                    li = sym - 257
                    length = LENGTH_BASE[li] + reader.read_bits(LENGTH_EXTRA[li])
                    dist_sym = _decode_huffman_symbol(reader, FIXED_DIST_TABLE)
                    distance = DIST_BASE[dist_sym] + reader.read_bits(DIST_EXTRA[dist_sym])
                    start = len(output) - distance
                    for k in range(length):
                        output.append(output[start + k])

        elif btype == 2:
            raise NotImplementedError("Dynamic Huffman blocks not supported in simplified PNG")
        else:
            raise ValueError(f"Invalid block type: {btype}")

        if bfinal:
            break

    return bytes(output)


def _adler32(data: bytes) -> int:
    a = 1
    b = 0
    MOD = 65521
    for byte in data:
        a = (a + byte) % MOD
        b = (b + a) % MOD
    return (b << 16) | a


def zlib_compress(data: bytes) -> bytes:
    cmf = 0x78
    flg = 0x01
    compressed = deflate_compress(data)
    checksum = _adler32(data)
    return (struct.pack('BB', cmf, flg)
            + compressed
            + struct.pack('>I', checksum))


def zlib_decompress(data: bytes) -> bytes:
    if len(data) < 6:
        raise ValueError("Invalid zlib stream: too short")
    cmf = data[0]
    flg = data[1]
    if (cmf * 256 + flg) % 31 != 0:
        raise ValueError("Invalid zlib stream: bad header check")
    cm = cmf & 0x0F
    if cm != 8:
        raise ValueError(f"Invalid zlib compression method: {cm}")
    checksum_stored = struct.unpack('>I', data[-4:])[0]
    compressed = data[2:-4]
    decompressed = deflate_decompress(compressed)
    checksum_computed = _adler32(decompressed)
    if checksum_computed != checksum_stored:
        raise ValueError(
            f"Adler-32 mismatch: stored=0x{checksum_stored:08X}, "
            f"computed=0x{checksum_computed:08X}"
        )
    return decompressed
