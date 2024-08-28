from io import BytesIO
import struct
from typing import overload
from .utils.io import BytesReader, BytesWriter

NXX_GZIP_MAGIC = b'NXGX'
NXX_DFLT_MAGIC = b'NXCX'

@overload
def nxx_decompress(src: str | BytesReader | bytes,
                   dest : None = None) -> BytesIO : ...
@overload
def nxx_decompress(src: str | BytesReader | bytes,
                   dest : str | BytesWriter) -> None : ...
def nxx_decompress(src: str | BytesReader | bytes,
                   dest : str | BytesWriter | None = None) :

    match src :
        case str() : file = open(src, 'rb')
        case bytes() | bytearray() | memoryview() : file = BytesIO(src)
        case _ : file = src
    magic = file.read(4)
    size, compressed_size, _ = struct.unpack("<III", file.read(12))

    data = file.read(compressed_size)
    assert len(data) == compressed_size

    if magic in NXX_GZIP_MAGIC :
        import gzip
        data = gzip.decompress(data)
    elif magic == NXX_DFLT_MAGIC :
        import zlib
        data = zlib.decompress(data)
    else :
        raise ValueError(f"Unexpected magic bytes {magic}")
    
    # TODO implement BNTX decompression
    
    match dest :
        case str() :
            with open(dest, 'wb') as file :
                file.write(data)
        case None :
            return BytesIO(data)
        case _ :
            dest.write(data)