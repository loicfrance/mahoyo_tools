# Original code from
# https://github.com/LinkOFF7/HunexFileArchiveTool
# Conversion to Python by requinDr and loicfrance

from __future__ import annotations
from io import BufferedRandom, BufferedReader, BufferedWriter, BytesIO
import os
import struct
from typing import Literal, Optional, Self, Union, cast, overload, TYPE_CHECKING


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
        """Get the HFA entry bytes data bytes
        
        If data is stored in the object, return the stored data.
        Otherwise, extract the data bytes from the file, and return it.
        Use `write()`, `loadData()` or `from_file()` methods, or set the \
        `data` attribute to store data bytes in the entry.

        :return: the entry data bytes
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
        if not isinstance(self._data, BytesIO) :
            self._data = BytesIO(self.data)
            self._data.seek(self._cursor)
        result = self._data.write(data)
        self._cursor = self._data.tell()
        if self._cursor > self._size :
            self._size = self._cursor
        return result
    
    def truncate(self, size: int | None = None, /) -> int:
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
        return self._cursor
    
    def seek(self, offset: int, whence: int = 0, /) -> int :
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
        """Write the entry data to the specified file
        
        :param dest: destination file path or writer
        """
    @overload
    def to_file(self, dest: None = None) -> BytesIO :
        """Write the entry data to a file
        
        :return: a file containing a copy of the entry data
        """
    def to_file(self, dest: BytesWriter | str | None = None):
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
    def extract(self, dest: str | BytesRW, *args, **kwargs) -> None : ...
    @overload
    def extract(self, dest: None = None, *args, **kwargs) -> BytesIO : ...

    def extract(self, dest: str | BytesRW | None = None, *args, **kwargs) :
        extension_index = self._file_name.rindex('.')
        extension = self._file_name[extension_index+1:]
        self.seek(0)
        match extension :
            case 'cbg' : return CompressedBG(self).img_write(dest, *args, **kwargs)
            case 'ctd' :
                lenzu_header = self.read(16)
                self.seek(0)
                if lenzu_header == b"LenZuCompressor\0" :
                    return lenzu_decompress(self, self.size, dest)
                return self.to_file(dest)
    
    def inject(self, src: str | BytesReader | bytes) :
        if not isinstance(self._data, BytesIO) :
            self.loadData()
        extension_index = self._file_name.rindex('.')
        extension = self._file_name[extension_index+1:]
        self.seek(0)
        match extension :
            case 'cbg' :
                CompressedBG(self).img_read(src)
            case 'ctd' :
                if isinstance(src, str) :
                    self.from_file(src)
                else :
                    self.data = src
    
#endregion #####################################################################
#region                            HFA ARCHIVE
################################################################################

class HfaArchive :

    _path: str
    _file: BufferedRandom | None
    _entries: dict[str, _HfaEntry]

    def __init__(self, path: str, mode: Literal['r', 'rw'] = 'r') -> None:
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

    def __exit__(self, *error) :
        self.close()

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
                i = 2
                while f"{key}({i})" in self._entries :
                    i += 1
                key = f"{key}({i})"
            self._entries[key] = e

    def close(self) :
        """Close the archive file, and clear all entries"""
        if self._file is not None :
            if self.mode == "rw" :
                self.hfa_write()
            self._file.close()
            self._file = None
            self._entries = {}

    def hfa_write(self, dest: Union[BufferedWriter, str, None] = None,
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
            file.write(entry.read())