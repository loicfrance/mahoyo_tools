# Original code from
# https://github.com/LinkOFF7/HunexFileArchiveTool
# Conversion to Python by requinDr and loicfrance

from __future__ import annotations
from io import BufferedRandom, BufferedReader, BufferedWriter, BytesIO
import os
import struct
from typing import Literal, Self, cast, overload, TYPE_CHECKING

from .mzp import MzpImage
from .lenzu import lenzu_decompress
from .cbg import CompressedBG
from .utils.io import BytesReader, BytesRW, BytesWriter
if TYPE_CHECKING :
    from _typeshed import ReadableBuffer

__all__ = [
    "hfa"
]

HFA_FILE_MAGIC = b"HUNEXGGEFA10"
HFA_HEADER_SIZE = len(HFA_FILE_MAGIC) + 4
HFA_ENTRY_HEADER_SIZE = 0x80

def hfa(path: str, mode: Literal['r', 'rw'] = 'r') :
    return HfaArchive(path, mode)

################################################################################
#region                             HFA ENTRY
################################################################################

class _HfaEntry :
    def __init__(self, hfa: HfaArchive, index: int) -> None:
        file = hfa._file
        assert file is not None
        pos = file.tell()
        file.seek(HFA_HEADER_SIZE + index * HFA_ENTRY_HEADER_SIZE)
        self._hfa = hfa
        self._key = None
        self._data = None
        self._file_name = file.read(0x60).rstrip(b'\0').decode('utf-8')
        self._data_offset, self._size, *self.reversed = struct.unpack("<8I", file.read(8*4))
        file.seek(pos)
        self._cursor = 0

#__________________________________properties___________________________________
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def index(self) -> int :
        """Index of the entry in the hfa file"""
        return list(self._hfa._entries.values()).index(self)

    @property
    def key(self) -> str :
        """Key of the entry in the HfaArchive _entries dictionary"""
        return self._key

    @property
    def name(self) -> str :
        """Name of the file stored in the hfa entry"""
        return self._file_name

    @property
    def header_start(self) -> int :
        """Address of the entry header in the hfa file"""
        return HFA_HEADER_SIZE + self.index * HFA_ENTRY_HEADER_SIZE

    @property
    def data_start(self) -> int :
        """Address of the entry data in the hfa file"""
        return self._hfa._data_start_offset + self._data_offset

    @property
    def data_end(self) -> int :
        """Address of the end of entry data in the hfa file"""
        return self.data_start + self._size

    @property
    def size(self) -> int :
        """Size (in bytes) of the entry data"""
        return self._size
    
    @property
    def header(self) -> bytes :
        """Header bytes of the entry"""
        return self.name.encode("utf-8").ljust(0x60, b'\0') + \
            struct.pack("<8I", self._data_offset, self.size, *self.reversed)
    
    @property
    def data(self) -> bytes :
        """Raw data bytes of the entry"""
        match self._data :
            case None :
                pos = self.tell()
                self.seek(0)
                data = self.read()
                self.seek(pos)
                return data
            case BytesIO() :
                return self._data.getvalue()
            case _ :
                return self._data
    @data.setter
    def data(self, data: bytes | BytesReader) :
        match data :
            case bytes() : self._data = data
            case _ :
                assert not isinstance(data, (memoryview, bytearray))
                self._data = data.read()
        self._size = len(self._data)

#________________________________special methods________________________________
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __repr__(self) :
        reversed_str = ','.join(map(hex, self.reversed))
        return f"HFA_entry: {self.name}, " + \
            f"offset = 0x{self._data_offset:08x}, " + \
            f"size = 0x{self.size:08x}, " + \
            f"reversed = {reversed_str}"

#_______________________________file-like methods_______________________________
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def read(self, size: int|None = None, /) -> bytes :
        """Read bytes from the entry as a file
        
        If the size if not specified, the remaining bytes to the end of the
        entry are returned.
        An internal cursor is used to keep track of where the last read / write
        instruction ended, and start the new one at this position.
        :param size: number of bytes to read. None or unspecified to read all \
                     remaining bytes
        :return: the bytes read from the entry
        """
        if isinstance(self._data, BytesIO) :
            data = self._data.read(size)
            self._cursor = self._data.tell()
            return data
        if size is None :
            size = self.size - self._cursor
        elif size + self._cursor > self.size :
            size = self._size - self._cursor
        if self._data is None :
            file = self._hfa._file
            assert file is not None
            pos = file.tell()
            file.seek(self.data_start + self._cursor)
            data = file.read(size)
            file.seek(pos)
        else :
            data = self._data[self._cursor : self._cursor + cast(int, size)]
        self._cursor += cast(int, size)
        return data

    def write(self, data: 'ReadableBuffer', /) -> int :
        """Write bytes to the entry as a file
        
        An internal cursor is used to keep track of where the last read / write
        instruction ended, and start the new one at this position.
        :param data: buffer containing the data bytes to write
        :return: number of bytes written to the entry
        """
        if not isinstance(self._data, BytesIO) :
            self._data = BytesIO(self.data)
            self._data.seek(self._cursor)
        result = self._data.write(data)
        self._cursor = self._data.tell()
        if self._cursor > self._size :
            self._size = self._cursor
        return result
    
    def truncate(self, size: int | None = None, /) -> int:
        """Change the size of the entry
        
        If the size is lower than the current size, the extra bytes are removed.
        If the size is greater than the current size, 0x00 bytes are appended.
        If the size is not specified, the content is cut at the current \
        internal cursor location (where the last read/write instruction \
        stopped).
        :param size: new desired size of the entry. None or unspecified to cut \
            at the current position
        :return: the new entry size
        """
        if size is None :
            size = self._cursor
        if size > self._size :
            if not isinstance(self._data, BytesIO) :
                self._data = BytesIO(self.data)
                self._data.truncate(size)
        else :
            match self._data :
                case None : pass
                case bytes() : self._data = self._data[:size]
                case BytesIO() : self._data.truncate(size)
            if self._cursor > size :
                self._cursor = size
        self._size = size
        return size
    
    def tell(self) :
        """Get the current internal cursor position in the entry data
        
        This cursor is where the last read/write instruction
        stopped.
        :return: the internal cursor position"""
        return self._cursor
    
    def seek(self, offset: int, whence: int = 0, /) -> int :
        """Change the position of the internal cursor
        
        :param offset: position of the cursor, relative to the start, current \
            position or end depending on the whence parameter
        :param whence: `0` to position the cursor relative to the start, \
            `1` to position the cursor relative to the current position, and \
            `2` to position the cursor relative to the end; Defaults to `0`
        :return: new cursor position relative to the start
        """
        size = self.size
        match self._data :
            case bytes() | None :
                match whence :
                    case 0 : self._cursor = offset
                    case 1 : self._cursor += offset
                    case 2 : self._cursor = size + offset
                    case _ : raise ValueError(f"whence = {{0 | 1 | 2}} (found {whence})")
                assert 0 <= self._cursor <= size
            case _ :
                assert not isinstance(self._data, (memoryview, bytearray))
                self._cursor = self._data.seek(offset, whence)
        return self._cursor

#_________________________________other methods_________________________________
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def loadData(self) :
        """Read the data bytes from the file and store it in the entry
        
        The stored data will be used instead of reading (again) from the \
        source file when calling the `read()` method
        """
        self.data = self.data
    
    @overload
    def to_file(self, dest: str | BytesWriter) -> None :
        """Write the raw entry data to the specified file
        
        :param dest: destination file path or writer
        """
    @overload
    def to_file(self, dest: None = None) -> BytesIO :
        """Write the rawentry data to a file
        
        :return: a file containing a copy of the entry data
        """
    def to_file(self, dest: BytesWriter | str | None = None):
        self.seek(0)
        match dest :
            case None :
                return BytesIO(self.read())
            case str() :
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.path.isdir(os.path.abspath(dest)) :
                    dest = os.path.join(dest, self.name)
                file = open(dest, "rb+" if os.path.exists(dest) else "wb+")
                file.write(self.read())
                file.close()
            case _ :
                dest.write(self.read())

    def from_file(self, src: BytesReader|str) :
        """Read the entry data from the specified file
        
        Read all bytes from the source file and store it in the entry.

        :param src: source file path or reader, from which the data will be read
        """
        if isinstance(src, str) :
            if os.path.isdir(os.path.abspath(src)) :
                src = os.path.join(src, self.name)
            file = open(src, "rb")
            self.data = file
            file.close()
        else :
            self.data = src
    
    @overload
    def extract(self, dest: str | BytesRW, *args, **kwargs) -> None :
        """Extract the content of the entry to the specified location.
        
        If the destination is file path that has the same extension \
        as the entry, the raw data is inserted at the specified location.
        Otherwise, the file content is decompressed (if possible) and copied \
        into the destination file.

        :param dest: destination file path or writable file-like object
        :param args: additional arguments for the decompression
        :param kwargs: additional arguments for the decompression
        """
    @overload
    def extract(self, dest: None = None, *args, **kwargs) -> BytesIO :
        """Extract the content of the entry and store it in a `BytesIO` object

        The file content is decompressed (if possible) and copied \
        into the created `BytesIO` object.

        :param args: additional arguments for the decompression
        :param kwargs: additional arguments for the decompression
        """

    def extract(self, dest: str | BytesRW | None = None, *args, **kwargs) :
        extension_index = self._file_name.rindex('.')
        extension = self._file_name[extension_index+1:]
        self.seek(0)
        if isinstance(dest, str) and dest[dest.rfind('.')+1:] == extension :
            self.to_file(dest)
        else :
            match extension :
                case 'cbg' :
                    cbg = CompressedBG(self)
                    return cbg.img_write(dest, *args, **kwargs)
                case 'ctd' :
                    lenzu_header = self.read(16)
                    self.seek(0)
                    if lenzu_header == b"LenZuCompressor\0" :
                        return lenzu_decompress(self, self.size, dest)
                    return self.to_file(dest)
                case 'mzp' :
                    mzpImg = MzpImage(self.data)
                    return mzpImg.img_write(dest)
                case 'mp4' | 'chs' | 'ccit' | 'hw' :
                    return self.to_file(dest)

    def inject(self, src: str | BytesReader | bytes, **kwargs) :
        """Inject a new content to the entry
        
        If the source is a file path that has the same extension as the entry, \
        the raw bytes are read from the file and inserted in the entry.
        Otherwise, the source file content is compressed (if possible) and \
        copied to the entry.
        """
        if self._data is None :
            self.loadData()
        extension_index = self._file_name.rindex('.')
        extension = self._file_name[extension_index+1:]
        self.seek(0)
        if isinstance(src, str) and src[src.rfind('.')+1:] == extension :
            self.from_file(src)
        else :
            match extension :
                case 'cbg' :
                    CompressedBG(self).img_read(src)
                case 'ctd' | 'mp4' | 'chs' | 'ccit' | 'hw' :
                    if isinstance(src, str) :
                        self.from_file(src)
                    else :
                        self.data = src
                case 'mzp' :
                    mzpImg = MzpImage(self.data)
                    mzpImg.img_read(src, **kwargs)
                    self.data = mzpImg.mzp_write()
    
#endregion #####################################################################
#region                            HFA ARCHIVE
################################################################################

class HfaArchive :

    _path: str
    _file: BufferedRandom | None
    _entries: dict[str, _HfaEntry]

    def __init__(self, path: str, mode: Literal['r', 'rw'] = 'r') -> None:
        """HFA archive object
        
        :param path: path to the `.hfa` file to read
        :param mode: 'rw' to replace the file content when caling `close()` , \
            'r' (default) otherwise
        """
        self._path = path
        self._file = None
        self._mode: Literal['r', 'rw'] = mode
        self._entries = { }

#__________________________________properties___________________________________
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @property
    def mode(self) -> Literal['r', 'rw'] :
        return self._mode

    @property
    def path(self) -> str :
        """Path to the `.hfa` archive file"""
        return self._path

    @property
    def _data_start_offset(self) -> int :
        """Position of the first entry's data in the `.hfa` file"""
        return HFA_HEADER_SIZE + len(self) * HFA_ENTRY_HEADER_SIZE

#________________________________special methods________________________________
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __contains__(self, key: int | str) -> bool :
        match key :
            case int() : return key < len(self._entries)
            case str() : return key in self._entries

    def __getitem__(self, key: int | str) -> _HfaEntry :
        match key :
            case int() : return list(self._entries.values())[key]
            case str() : return self._entries[key]

    def __iter__(self) :
        assert self._file is not None and not self._file.closed
        yield from self._entries.values()

    def __len__(self) :
        return len(self._entries)
    
    def __enter__(self) -> Self :
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) :
        self.close(exc_type, exc_value, traceback)

#____________________________________methods____________________________________
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def entries(self) :
        """Return a view on the archive entries.
        
        :return: an list-like object that references the archive entries"""
        return self._entries.values()

    def open(self) :
        """Open the archive file, and extract entry details
        
        Open the `.hfa` archive file with the `rb+` mode, check if this is a \
        valid HFA archive, then extract all entry details, without extracting
        entry data.
        """
        if self._file is None :
            self._file = open(self._path, "rb+")
        header = self._file.read(HFA_HEADER_SIZE)
        assert header.startswith(HFA_FILE_MAGIC)
        nbEntries = int.from_bytes(header[-4:], 'little', signed=False)
        entries = [_HfaEntry(self, i) for i in range(0, nbEntries)]
        self._entries = {}
        for e in entries :
            key = e._file_name
            if key in self._entries :
                try :
                    base, ext = key.rsplit('.', 1)
                    i = 2
                    while f"{base}({i}).{ext}" in self._entries :
                        i += 1
                    key = f"{base}({i}).{ext}" if ext else f"{base}({i})"
                except ValueError :
                    i = 2
                    while f"{key}({i})" in self._entries :
                        i += 1
                    key = f"{key}({i})"
            e._key = key
            self._entries[key] = e

    def close(self, exc_type = None, exc_value = None, traceback = None) :
        """Close the archive file, and clear all entries
        
        If the archive is opened in `rw` mode, the original file content is \
        replaced with the new content.
        """
        if self._file is not None :
            if self.mode == "rw" and exc_type is None :
                self.hfa_write()
            self._file.close()
            self._file = None
            self._entries = {}

    def hfa_write(self, dest: BufferedWriter | str | None = None,
                  align_size: int = 0, align_byte: int = 0x00) :
        """Write the hfa content to the original file, or to the specified file
        
        Build and write the `.hfa` archive to the specified file, or to the \
        source `.hfa` file if no destination file is provided.
        :param dest: destination file path of writer. If `None` or not \
            provided, the source file is written instead
        :param align_size: pad between entries to align entry offsets to \
            the specified alignment. Default to `0` (no alignment)
        :param align_bytes: byte-size int used when padding after entry data. \
            Default to `0x00`
        """

        assert self._file is not None and not self._file.closed
        match dest :
            case str() : file = open(dest, "wb")
            case None : file = self._file
            case _ : file = dest

        file.seek(0)
        file.write(HFA_FILE_MAGIC)
        file.write(len(self).to_bytes(4, 'little', signed=False))
        offset = 0
        for entry in self :
            if entry._data_offset != offset :
                if entry._data is None :
                    entry.loadData()
                entry._data_offset = offset
            file.write(entry.header)
            offset += entry.size
            if align_size > 0 :
                padding = align_size - offset % align_size
                if padding < align_size :
                    offset += padding
        for entry in self :
            pos = file.tell()
            start = entry.data_start
            if align_size > 0 :
                padding = start - pos
                assert padding < align_size
                if padding > 0 :
                    file.write(align_byte.to_bytes(1)*padding)
            else :
                assert pos == start
            entry.to_file(file)