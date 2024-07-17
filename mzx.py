
from io import BufferedRandom, BufferedReader, BytesIO
from math import ceil
from typing import Literal, Optional, Union, cast, overload
import numpy as np

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
    
    def append(self, buffer: bytes) -> None :
        pos = self._file.tell()
        
        if pos + len(buffer) <= self._size :
            self._file.write(buffer)
        else :
            split_index = self._size - pos
            self._file.write(buffer[:split_index])
            self._file.seek(0)
            self._file.write(buffer[split_index:])
    
    def get(self, index: int, size: int) -> bytes :
        pos = self._file.tell()
        self._file.seek(index)
        result = self._file.read(size)
        self._file.seek(pos)
        return result
    
    def peek_back(self, size: int) -> bytes :
        pos = self._file.tell()
        read_pos = pos - size
        if read_pos >= 0 :
            self._file.seek(read_pos)
            return self._file.read(size)
        else :
            self._file.seek(read_pos, 2)
            part1 = self._file.read(-read_pos)
            self._file.seek(0)
            return part1 + self._file.read(pos)

def rle_compress_length(words: np.ndarray, cursor: int, clear_count: int,
                        invert_bytes: bool) :
    
    word = words[cursor]
    if clear_count == 0x1000 :
        last = 0xFFFF if invert_bytes else 0x0000
    else :
        last = words[cursor-1]
    if word == last :
        max_chunk_size = min(64, words.size - cursor)
        length = 1
        while length < max_chunk_size and words[cursor+length] == last :
            length += 1
        return length
    else :
        return 0

def backref_compress_length(words: np.ndarray, cursor: int) :
    start = max(cursor - 256, 0)
    occurrences = np.where(words[start:cursor] == words[cursor])
    best_index = -1
    best_len = 0
    for o in occurrences[0] :
        length = 1
        back_ref = words[start:start+64] # no need to stop at cursor
        while o+length < back_ref.size and \
                cursor + length < words.size and \
                back_ref[o+length] == words[cursor+length] :
            length += 1
        if length > best_len :
            best_index = o
            best_len = length
    if best_len > 0 :
        distance = cursor - (start + best_index)
        return (distance, best_len)
    else :
        return (0, 0)

@overload
def mzx_compress(src: BytesReader, dest: BytesWriter,
                 invert_bytes: bool = False, level: int = 0) -> None : ...
@overload
def mzx_compress(src: BytesReader, dest: None = None,
                 invert_bytes: bool = False, level: int = 0) -> BytesIO : ...
def mzx_compress(src: BytesReader, dest: BytesWriter | None = None,
                 invert_bytes: bool = False, level: int = 0) :
    start = src.tell()
    end = src.seek(0, 2)
    src.seek(start)
    match dest :
        case None : output_file = BytesIO()
        case _ : output_file = dest
    header = MZX_FILE_MAGIC + (end - start).to_bytes(4, 'little', signed=False)
    output_file.write(header)
    words = np.frombuffer(src.read(), dtype=np.uint8)
    if words.size % 2 == 1 :
        words = np.append(words, 0x00)
    words = words.view(dtype="<u2")
    end = len(words)
    if level == 0 :
        cursor = 0
        while cursor < end :
            # Len field is 6 bits, each word is 2 bytes,
            # we can write 64 words (128 bytes) per literal record
            chunk_size = min(end - cursor, 64)

            cmd: int = (MzxCmd.LITERAL | ((chunk_size - 1) << 2))
            output_file.write(cmd.to_bytes(1, 'little'))
            output_file.write(words[cursor:cursor+chunk_size])
            cursor += chunk_size
    elif level == 1 :
    
        clear_count = 0
        #ring_buf = RingBuffer(128, 0xFF if invert_bytes else 0x00)
        literal_start = 0
        literal_len = 0
        best_len = 0
        cursor = 0
        while cursor < words.size :
            #print(f"{cursor}/{words.size}", end="\r")
            
            if clear_count <= 0:
                clear_count = 0x1000
            if cursor > 0 :
                rle_len = rle_compress_length(words, cursor, clear_count, invert_bytes)
                best_len = rle_len

                if best_len < 64 :
                    # back-ref
                    br_dist, br_len = backref_compress_length(words, cursor)
                    if br_len > best_len :
                        best_len = br_len
                    # TODO implement ring-buffer
                else :
                    br_len = 0


                if literal_len > 0 and best_len == 1 : # TODO maybe <= 2 ?
                    best_len = 0 # changing from literal is not worth it
            else :
                best_len = 0
            if best_len == 0 :
                # best is literal
                if literal_len == 0 :
                    literal_start = cursor
                    literal_len = 1
                else :
                    literal_len += 1
                cursor += 1
                clear_count -= 1
            else :
                while literal_len > 0 :
                    length = min(literal_len, 64)
                    cmd = (MzxCmd.LITERAL | ((length - 1) << 2))
                    output_file.write(cmd.to_bytes(1))
                    chunk = words[literal_start:literal_start+length]
                    if invert_bytes :
                        chunk ^= 0xFFFF
                    output_file.write(chunk.tobytes())
                    literal_start += length
                    literal_len -= length
                if best_len == rle_len :
                    cmd = (MzxCmd.RLE | ((rle_len - 1) << 2))
                    output_file.write(cmd.to_bytes(1))
                    clear_count -= rle_len
                elif best_len == br_len :
                    cmd = (MzxCmd.BACKREF | (br_len - 1) << 2)
                    output_file.write(cmd.to_bytes(1))
                    output_file.write(int(br_dist-1).to_bytes(1))
                    clear_count -= br_len
                else :
                    assert False, "Should never reach this point"
                cursor += best_len
        
        if literal_len > 0 :
            while literal_len > 0 :
                length = min(literal_len, 64)
                literal_len -= length
                cmd = (MzxCmd.LITERAL | ((length - 1) << 2))
                output_file.write(cmd.to_bytes(1))
                chunk = words[literal_start:literal_start+length]
                if invert_bytes :
                    chunk ^= 0xFFFF
                output_file.write(chunk.tobytes())
                literal_start += length
                literal_len -= length
    else :
        raise ValueError(f"MZX compresison level {level} not implemented")

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