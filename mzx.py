

from io import BufferedRandom, BufferedReader, BytesIO
from math import ceil
from typing import Optional, Union, cast, overload

from .utils.io import BytesRW, BytesReader, BytesWriter

MZX_FILE_MAGIC = b"MZX0"

class MzxCmd :
    RLE = 0
    BACKREF = 1
    RINGBUF = 2
    LITERAL = 3

class RingBuffer :
    def __init__(self, size: int, baseValue: int) -> None:
        self._file = BytesIO(bytes([baseValue]) * size)
        self._size = size
    
    def append(self, buffer: bytes) :
        pos = self._file.tell()
        
        if pos + len(buffer) <= self._size :
            self._file.write(buffer)
        else :
            split_index = self._size - pos
            self._file.write(buffer[:split_index])
            self._file.seek(0)
            self._file.write(buffer[split_index:])
    
    def get(self, index: int, size: int) :
        pos = self._file.tell()
        self._file.seek(index)
        result = self._file.read(size)
        self._file.seek(pos)
        return result

@overload
def mzx_compress(src: BytesReader, dest: BytesWriter,
                 invert_bytes: bool = False) -> None : ...
@overload
def mzx_compress(src: BytesReader, dest: None = None,
                 invert_bytes: bool = False) -> BytesIO : ...
def mzx_compress(src: BytesReader, dest: BytesWriter | None = None,
                 invert_bytes: bool = False) :
    start = src.tell()
    end = src.seek(0, 2)
    src.seek(start)
    match dest :
        case None : output_file = BytesIO()
        case _ : output_file = dest
    header = MZX_FILE_MAGIC + (end - start).to_bytes(4, 'little', signed=False)
    output_file.write(header)
    while src.tell() < end :
        # Len field is 6 bits, each word is 2 bytes,
        # we can write 128 bytes per literal record
        bytes_remaining = end - src.tell()
        bytes_to_write = min(bytes_remaining, 128)

        # Convert to a number of words to write.
        # If we have a trailing byte, that needs an extra word
        words_to_write = ceil(bytes_to_write / 2)

        # The number of words to write offset by 1 in the literal
        words_to_write -= 1

        # Pack the cmd
        cmd: int = (MzxCmd.LITERAL | (words_to_write << 2)) & 0xFF
        output_file.write(cmd.to_bytes(1, 'little'))

        words = src.read(bytes_to_write)

        if invert_bytes :
            words = bytes(map(lambda byte : byte ^ 0xFF, words))
        output_file.write(words)
    output_file.seek(0)
    src.seek(start)
    if dest is None :
        return output_file

@overload
def mzx_decompress(src: BytesReader | str, dest: BytesRW | str,
                   invert_bytes: bool = False) -> None : ...

@overload
def mzx_decompress(src: Union[BytesReader, str],
                   dest: None = None,
                   invert_bytes: bool = False) -> BytesIO : ...

def mzx_decompress(src: BytesReader | str,
                   dest: BytesRW | str | None = None,
                   invert_bytes: bool = False) :
    input_file = open(src, 'rb') if isinstance(src, str) else src
    match dest :
        case str() : output_file = open(dest, "wb+")
        case None : output_file = BytesIO()
        case _ : output_file = dest; output_file.seek(0)
    start = input_file.tell()
    end = input_file.seek(0, 2)
    input_file.seek(start)

    len_header = len(MZX_FILE_MAGIC) + 4
    header = input_file.read(len_header)
    assert len(header) == len_header
    assert header[:len(MZX_FILE_MAGIC)] == MZX_FILE_MAGIC
    
    decompressed_size = int.from_bytes(header[-4:], 'little', signed = False)

    output_file.truncate(decompressed_size)

    filler_2bytes = b'\xFF\xFF' if invert_bytes else b'\0\0'

    ring_buf = RingBuffer(128, 0xFF if invert_bytes else 0)

    clear_count = 0

    while output_file.tell() < decompressed_size and input_file.tell() < end:
        # Get type / arg from next byte in input
        flags = input_file.read(1)[0]
        cmd = flags & 0x03
        arg = flags >> 2

        if clear_count <= 0:
            clear_count = 0x1000

        match cmd :
            case MzxCmd.RLE :
                # Repeat last two bytes arg + 1 times
                if clear_count == 0x1000 :
                    last = filler_2bytes
                else :
                    output_file.seek(-2, 1)
                    last = output_file.read(2)
                output_file.write(last * (arg + 1))

            case MzxCmd.BACKREF :
                # Next byte from input is lookback distance - 1, in 2-byte words
                pos = output_file.tell()
                k = 2 * (input_file.read(1)[0] + 1)
                length = 2 * (arg + 1)
                output_file.seek(pos-k)
                buffer = output_file.read(length)
                if k < length :
                    buffer = (buffer * ceil(length/k))[:length]
                output_file.seek(pos)
                output_file.write(buffer)

            case MzxCmd.RINGBUF :
                # Write ring buffer data at position arg
                output_file.write(ring_buf.get(arg*2, 2))
            
            case _ : # 3: LITERAL
                buffer = input_file.read((arg+1)*2)
                if invert_bytes :
                    buffer = bytes([b ^ 0xFF for b in buffer])
                output_file.write(buffer)
                ring_buf.append(buffer)
        
        clear_count -= 1 if cmd == MzxCmd.RINGBUF else arg + 1

    output_file.truncate(decompressed_size)  # Resize stream to decompress size
    output_file.seek(0)
    input_file.seek(start)

    if isinstance(src, str) : cast(BufferedReader, input_file).close()
    if isinstance(dest, str) : cast(BufferedRandom, output_file).close()
    elif dest is None : return output_file