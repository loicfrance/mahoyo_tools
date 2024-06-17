# Original code from
# https://github.com/LinkOFF7/HunexFileArchiveTool
# Conversion to Python by requinDr
from __future__ import annotations
from io import BufferedRandom, BufferedReader, BufferedWriter, BytesIO
import os
from typing import Optional, Union, overload

__all__ = [
    "HfaArchive"
]

HFA_FILE_MAGIC = b"HUNEXGGEFA10"
HFA_HEADER_SIZE = len(HFA_FILE_MAGIC) + 4
HFA_ENTRY_HEADER_SIZE = 0x80

class HfaArchiveEntry :
    def __init__(self, hfa: HfaArchive, index: int) -> None:
        file = hfa._file
        pos = file.tell()
        file.seek(HFA_HEADER_SIZE + index * HFA_ENTRY_HEADER_SIZE)
        self._hfa = hfa
        self._data = None
        self.file_name = file.read(0x60).rstrip(b'\0').decode('utf-8')
        self.hfa_offset = int.from_bytes(file.read(4), 'little')
        self._size = int.from_bytes(file.read(4), 'little')
        self.reversed = [
            int.from_bytes(file.read(4), byteorder='little')
            for _ in range(6)
        ]
        file.seek(pos)
    
    @property
    def index(self) -> int :
        """Index of the entry in the hfa file"""
        return self._hfa._entries.index(self)
    
    @property
    def header_start(self) -> int :
        """Address of the entry header in the hfa file"""
        return HFA_HEADER_SIZE + self.index * HFA_ENTRY_HEADER_SIZE
    
    @property
    def data_start(self) -> int :
        """Address of the entry data in the hfa file"""
        return self._hfa.data_start_offset + self.hfa_offset
    
    @property
    def data_end(self) -> int :
        """Address of the end of entry data in the hfa file"""
        return self.data_start + self._size
    
    @property
    def size(self) -> int :
        """Size (in bytes) of the entry data"""
        return self._size
    
    @property
    def data(self) -> bytes :
        """Entry data"""
        if self._data is None :
            file = self._hfa._file
            pos = file.tell()
            file.seek(self.data_start)
            data = file.read(self.size)
            file.seek(pos)
            return data
        else :
            return self._data
    
    @data.setter
    def data(self, data: bytes|BufferedReader) :
        if not isinstance(data, bytes) :
            data = data.read()
        self._data = data
        self._size = len(data)

    def loadData(self) :
        """Store the data of the entry in the HfaArchiveEntry object"""
        self.data = self.data

    def header(self) -> bytes :
        """Create the entry's header bytes"""
        reversed_bytes = b''.join([
            x.to_bytes(4, 'little', signed=False)
            for x in self.reversed
        ])
        return \
            self.file_name.encode("utf-8").ljust(0x60, b'\0') + \
            self.hfa_offset.to_bytes(4, 'little', signed=False) + \
            self.size.to_bytes(4, 'little', signed=False) + \
            reversed_bytes
    @overload
    def to_file(self, dest: str) -> None : ...
    @overload
    def to_file(self, dest: Optional[BytesIO] = None) -> BytesIO : ...

    def to_file(self, dest: Union[BytesIO, str, None] = None) -> Optional[BytesIO]:
        """Write the entry data to the specified file"""
        if dest is None :
            return BytesIO(self.data)
        elif isinstance(dest, str) :
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.isdir(os.path.abspath(dest)) :
                dest = os.path.join(dest, self.file_name)
            file = open(dest, "rb+" if os.path.exists(dest) else "wb+")
            file.write(self.data)
            file.close()
        elif dest is not None:
            pos = dest.tell()
            dest.write(self.data)
            dest.seek(pos)
            return dest
    
    def from_file(self, src: Union[BufferedReader, str]) :
        """Read the entry data from the specified file"""
        if isinstance(src, str) :
            if os.path.isdir(os.path.abspath(src)) :
                src = os.path.join(src, self.file_name)
            file = open(src, "rb")
            self.data = file.read()
            file.close()
        else :
            self.data = src.read()
            
    def __repr__(self) :
        reversed_str = ','.join(map(lambda x: f"0x{x:x}", self.reversed))
        return f"HFA_entry: {self.file_name}, " + \
            f"offset = 0x{self.hfa_offset:08x}, " + \
            f"size = 0x{self.size:08x}, " + \
            f"reversed = {reversed_str}"
    
class HfaArchive :

    _path: str
    _file: Union[BufferedRandom, None]
    _entries: list[HfaArchiveEntry]

    def __init__(self, path: str) -> None:
        self._path = path
        self._file = None
        self._entries = []
    
    def __getitem__(self, key: Union[int, str]) -> HfaArchiveEntry :
        if isinstance(key, int) :
            return self._entries[key]
        else :
            for e in self._entries :
                if e.file_name == key :
                    return e
            raise KeyError(f"unknonw entry {key}")
    
    def __contains__(self, key: str) -> bool :
        for e in self._entries :
            if e.file_name == key :
                return True
        return False
    
    def __iter__(self) :
        assert self.isOpen()
        yield from self._entries
    
    def __enter__(self) -> HfaArchive:
        self.open()
        return self

    def __exit__(self, *error) :
        self.close()
    
    @property
    def nb_entries(self) -> int :
        """number of entries in the hfa file"""
        assert self.isOpen()
        return len(self._entries)
    
    @property
    def data_start_offset(self) -> int :
        return HFA_HEADER_SIZE + self.nb_entries * HFA_ENTRY_HEADER_SIZE

    def isOpen(self) -> bool :
        return self._file is not None
    
    def open(self) :
        if self._file is None :
            if os.path.exists(self._path) :
                self._file = open(self._path, "rb+")
            else :
                self._file = open(self._path, "wb+")
        header = self._file.read(HFA_HEADER_SIZE)
        assert header.startswith(HFA_FILE_MAGIC)
        nbEntries = int.from_bytes(header[-4:], 'little', signed=False)
        self._entries = [HfaArchiveEntry(self, i) for i in range(0, nbEntries)]
    
    def close(self) :
        if self._file is not None :
            self._file.close()
            self._file = None
            self._entries = []
    
    def hfa_write(self, align_size: int = 0, align_byte: int = 0x00, *,
                  dest: Union[BufferedWriter, str, None] = None) :
        """Write the hfa content to the original file, or to the specified file"""
        offset = 0
        if dest is not None :
            if isinstance(dest, str) :
                file = open(dest, "wb")
            else :
                file = dest
        else :
            file = self._file

        assert self.isOpen()
        file.seek(0)
        file.write(HFA_FILE_MAGIC)
        file.write(self.nb_entries.to_bytes(4, 'little', signed=False))
        for entry in self :
            if entry.hfa_offset != offset :
                entry.loadData()
                entry.hfa_offset = offset
            file.write(entry.header())
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
            file.write(entry.data)
