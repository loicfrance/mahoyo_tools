from io import BytesIO
from math import ceil
import struct
from typing import overload, TYPE_CHECKING

from .utils.io import BytesReader, BytesRW
from .utils.huffman import IntHuffmanTable
from .utils.bitstream import BitStreamReader
if TYPE_CHECKING :
    from _typeshed import SupportsRead


LENZU_MAGIC = b"LenZuCompressor\0" + \
    b"\x31\0\0\0\x30\0\0\0\0\0\0\0\0\0\0\0"

def read_huffman_table(src: BytesReader, bitCount: int) :
    first_real_entry = 1 << bitCount
    index_bits = ceil(bitCount / 8)
    index_bytes = ceil(index_bits / 8)
    entries_to_fill = int.from_bytes(src.read(index_bytes), 'little')
    if entries_to_fill == 0 :
        entries_to_fill = first_real_entry

    # Indices are stored with weights only if it takes less bytes than
    # storing all weights including the zeros.
    if first_real_entry * 4 < (index_bits + 4) * entries_to_fill :
        weights = list(enumerate(struct.unpack_from(f"<{entries_to_fill}I", src.read(entries_to_fill*4))))
    else :
        weights = [
            (int.from_bytes(src.read(index_bytes), 'little'),
             int.from_bytes(src.read(4), 'little'))
        ]
    table = IntHuffmanTable(map(lambda x: x[0], weights), invert=True)
    for i, w in weights :
        table.getNode(i).weight = w
    
    max_entries = ((first_real_entry + 1) * first_real_entry) >> 1
    table.buildTree(max_entries)
    return table

def lenzu_crc(src: 'SupportsRead[bytes]', size: int,
              seed: int = 0, lutOffset: int = 0) :
    lut = [0x0e9, 0x115, 0x137, 0x1b1]
    crc = seed
    for i in range(0, size) :
        crc = ((crc + src.read(1)[0]) * lut[(lutOffset + i) & 3]) % (1 << 32)
    return crc

@overload
def lenzu_decompress(src: BytesReader, compressed_size: int,
                     dest: BytesRW | str) : ...
@overload
def lenzu_decompress(src: BytesReader, compressed_size: int,
                     dest: None = None) -> BytesIO : ...
def lenzu_decompress(src: BytesReader, compressed_size: int,
                     dest: BytesRW | str | None = None) :
    
    start = src.tell()

    # header
    magic = src.read(len(LENZU_MAGIC))
    assert magic == LENZU_MAGIC
    decomp_len, crc, _ = struct.unpack_from("<IQI", src.read(16))

    # decompressor options
    _, huffBcRaw, huffBcMin, brLowBcXUpper, brLowBc, brBaseDist \
         = struct.unpack_from('6B', src.read(6))
    assert 3 <= huffBcRaw < 16
    assert 3 <= huffBcMin < 16
    huffBitCount = max(huffBcRaw, huffBcMin)
    assert brLowBcXUpper < 16
    assert 0 <= brLowBc < brLowBcXUpper
    assert huffBitCount >= brLowBcXUpper - brLowBc
    assert 2 <= brBaseDist < 9
    
    huffTable = read_huffman_table(src, huffBitCount)
    match dest :
        case str() :
            output = open(dest, "wb+")
            output.truncate(decomp_len)
            output.seek(0)
        case None :
            output = BytesIO()
            output.truncate(decomp_len)
            output.seek(0)
        case _ :
            output = dest

    bitStream = BitStreamReader(src, msb_first=True)

    while output.tell() < decomp_len and src.tell() < start + compressed_size :
        isBackRef = bitStream.readBit() != 0
        length = huffTable.decodeSequence(bitStream)
        assert length >= 0
        if isBackRef :
            length += brBaseDist
            distanceHighBits = huffTable.decodeSequence(bitStream)
            assert distanceHighBits >= 0
            distanceLowBits = 0
            if brLowBc > 0 :
                distanceLowBits = bitStream.read(brLowBc)
            distance = distanceLowBits | (distanceHighBits << brLowBc)
            distance += brBaseDist
            pos = output.tell()
            output.seek(pos-distance)
            if length > distance :
                buffer = (output.read(distance) * ceil(length / distance))[:length]
            else :
                buffer = output.read(length)
            output.seek(pos)
            output.write(buffer)
        else :
            output.write(struct.pack(f"{length+1}B", *(
                    bitStream.read(8) for _ in range(0, length+1)
            )))
    assert output.tell() >= decomp_len
    output.truncate(decomp_len)
    output.seek(0)

    if False : # check CRC
        assert crc == lenzu_crc(output, decomp_len)
    if dest is None :
        return output
