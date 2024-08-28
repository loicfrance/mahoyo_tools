"""
Credits to mangetsu (https://github.com/rschlaikjer/mangetsu) for details on
archive files structure.
"""

################################################################################
#region                        imports, constants
################################################################################
from __future__ import annotations
from io import BytesIO
import os
import struct
from typing import BinaryIO, Iterable, Literal, overload

from .lenzu import lenzu_decompress
from .nxx import nxx_decompress
from .mzp import MzpImage
from .cbg import CompressedBG
from .utils.io import BytesRW, BytesReader, BytesWriter

NAM_ENTRY_LEN = 32
HED_ENTRY_LEN = 8
MRG_SECTOR_SIZE = 0x800

#endregion #####################################################################
#region                             HED, NAM
################################################################################

def readHED(path: str) :
    entries: list[tuple[int, int, int]] = []
    assert os.path.exists(path), f"file '{path}' does not exist"
    with open(path, 'rb') as file :
        size = file.seek(0, 2)
        assert size % HED_ENTRY_LEN == 0, \
            f"wrong size for hed file {path}: {size}"
        file.seek(0)
        nb_entries = round(size / HED_ENTRY_LEN)
        for _ in range(0, nb_entries) :
            buffer = file.read(HED_ENTRY_LEN)
            offset, size_sectors, size_decomp_sectors = struct.unpack("<IHH", buffer)
            if offset == 0xFFFF_FFFF :
                break
            entries.append((offset, size_sectors, size_decomp_sectors))
    return entries

def writeHED(path: str, entries:Iterable[tuple[int, int, int]]) :
    with open(path, 'wb') as file :
        for (offset, size_sectors, size_decomp_sectors) in entries :
            file.write(struct.pack("<IHH", offset, size_sectors, size_decomp_sectors))
        file.write(b'\xFF'*16)

def readNAM(path: str) :
    assert os.path.exists(path), f"file '{path}' does not exist"
    fileNames: list[str] = []
    with open(path, 'rb') as file :
        size = file.seek(0, 2)
        assert size % NAM_ENTRY_LEN == 0, \
            f"wrong size for nam file {path}: {size}"
        file.seek(0)

        nb_entries = round(size / NAM_ENTRY_LEN)
        for _ in range(0, nb_entries) :
            buffer = file.read(NAM_ENTRY_LEN)
            if buffer[-2:] == b'\r\n' :
                name = buffer.decode('utf-8').strip('\x00\r\n')
                fileNames.append(name)
            else :
                break
    
    return fileNames

def writeNAM(path: str, fileNames: Iterable[str]) :
    with open(path, 'wb') as file :
        for fileName in fileNames :
            buffer = fileName.encode('utf-8')
            padLength = NAM_ENTRY_LEN - 2 - len(buffer)
            assert padLength >= 0, \
                f"file name {fileName} too long (max {NAM_ENTRY_LEN - 2})"
            buffer = buffer + (b'\0' * padLength) + b'\r\n'
            assert len(buffer) == NAM_ENTRY_LEN, \
                f"error while inserting file name {fileName} in 'nam' file"
            file.write(buffer)
        assert file.tell() % NAM_ENTRY_LEN == 0, \
            f"error while writing to {path}: wrong file size {file.tell()}"

#endregion #####################################################################
#region                             MRG entry
################################################################################

class MrgEntry :
    
    def __init__(self, mrg: MrgArchive, name: str | None,
                 offset_sectors: int, size_comp_sectors: int,
                 size_decomp_sectors: int) :

        self._mrg = mrg
        self._name = name
        self._offset_sectors = offset_sectors
        self._size_comp_sectors = size_comp_sectors
        self._size_decomp_sectors = size_decomp_sectors
        self._data = None
    
    @property
    def compressed(self) :
        return self._size_decomp_sectors == self._size_comp_sectors
    
    @property
    def data(self) -> bytes :
        if self._data is not None :
            return self._data
        else :
            file = self._mrg._mrgFile
            assert file is not None, "mrg file is closed"
            pos = file.tell()
            file.seek(self._offset_sectors * MRG_SECTOR_SIZE)
            buffer = file.read(self._size_comp_sectors * MRG_SECTOR_SIZE)
            file.seek(pos)
            return buffer
    
    @data.setter
    def data(self, buffer: bytes) :
        self._data = buffer
        # TODO recalculate size_decomp and size_comp
    
    def from_file(self, src: BytesReader|str) :
        """Read the entry data from the specified file
        
        Read all bytes from the source file and store it in the entry.

        :param src: source file path or reader, from which the data will be read
        """
        if isinstance(src, str) :
            if os.path.isdir(os.path.abspath(src)) :
                assert self._name is not None, "file name not provided"
                src = os.path.join(src, self._name)
            with open(src, "rb") as file :
                self.data = file.read()
        else :
            self.data = src.read()

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
        data = self.data
        match dest :
            case None :
                return BytesIO(data)
            case str() :
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                if os.path.isdir(os.path.abspath(dest)) :
                    assert self._name is not None, "file name not provided"
                    dest = os.path.join(dest, self._name)
                with open(dest, "wb") as file :
                    file.write(data)
            case _ :
                dest.write(data)
        
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
        if self._name is not None :
            ext = self._name[self._name.rindex('.')+1:].lower()
            if isinstance(dest, str) and dest[dest.rfind('.')+1:].lower() == ext :
                self.to_file(dest)
                return
        else :
            ext = None
        match ext :
            case None | 'mp4' | 'chs' | 'ccit' | 'hw' :
                return self.to_file(dest)
            case 'cbg' :
                cbg = CompressedBG(self.data)
                return cbg.img_write(dest, *args, **kwargs)
            case 'ctd' :
                data = self.data
                lenzu_header = data[:16]
                if lenzu_header == b"LenZuCompressor\0" :
                    return lenzu_decompress(BytesIO(data), len(data), dest)
                else :
                    return self.to_file(dest)
            case 'mzp' :
                mzpImg = MzpImage(self.data)
                return mzpImg.img_write(dest)
            case 'nxz' :
                return nxx_decompress(self.data, dest)
            case _ :
                raise ValueError(f'Unimplemented extraction of {self._name}')

    def inject(self, src: str | BytesReader | bytes, **kwargs) :
        """Inject a new content to the entry
        
        If the source is a file path that has the same extension as the entry, \
        the raw bytes are read from the file and inserted in the entry.
        Otherwise, the source file content is compressed (if possible) and \
        copied to the entry.
        """
        if self._name is not None :
            ext = self._name[self._name.rindex('.')+1:].lower()
            if isinstance(src, str) and src[src.rfind('.')+1:].lower() == ext :
                self.from_file(src)
                return
        else :
            ext = None
        match ext :
            case None | 'ctd' | 'mp4' | 'chs' | 'ccit' | 'hw' :
                match src :
                    case str() :
                        self.from_file(src)
                    case bytes() :
                        self.data = src
                    case bytearray() | memoryview() :
                        self.data = bytes(src)
                    case _ :
                        self.from_file(src)
            case 'cbg' :
                CompressedBG(self.data).img_read(src)
            case 'mzp' :
                mzpImg = MzpImage(self.data)
                mzpImg.img_read(src, **kwargs)
                self.from_file(mzpImg.mzp_write())
            case _ :
                raise ValueError(f'Unimplemented injection to {self._name}')

#endregion #####################################################################
#region                            MRG archive
################################################################################

class MrgArchive :

    def __init__(self, path: str, mode: Literal['r', 'w'] = 'r') -> None:
        if path.endswith('.mrg') :
            self._baseName = path[:-4] # remove '.mrg' extension
        else :
            self._baseName = path
        self._mode = mode
        self._entries: list[MrgEntry] = []
        self._names: list[str] | None = None
        self._mrgFile: BinaryIO | None = None
    
    def _entry_name(self, index: int) -> str | None :
        if self._names is None :
            return None
        else :
            return self._names[index]
    
    def open(self) :
        mrgPath = f"{self._baseName}.mrg"
        namPath = f"{self._baseName}.nam"
        hedPath = f"{self._baseName}.hed"
        mode = "wb+" if self._mode == "w" else "rb"

        assert os.path.exists(mrgPath), f"file '{mrgPath}' does not exist"
        self._mrgFile = open(mrgPath, mode)

        entries_info = readHED(hedPath)

        self._entries.clear()
        if os.path.exists(namPath) :
            self._names = readNAM(namPath)
            assert len(self._names) == len(entries_info), \
                "unmatching number of entries in nam and hed files"
        else :
            self._names = None
        
        for i, (offset, size_comp, size_decomp) in enumerate(entries_info) :
            self._entries.append(MrgEntry(self, self._entry_name(i), offset,
                                          size_comp, size_decomp))
    
    def close(self) :
        if self._mrgFile is not None :
            self._mrgFile.close()
            self._mrgFile = None
    
#________________________________special methods________________________________
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def __contains__(self, key: int | str) -> bool :
        match key :
            case int() : return key < len(self._entries)
            case str() :
                assert self._names is not None, \
                    f"cannot check for file names (no '{self._baseName}.nam' file)"
                return key in self._names

    def __getitem__(self, key: int | str) -> MrgEntry :
        match key :
            case int() : return self._entries[key]
            case str() :
                assert self._names is not None, \
                    f"cannot check for file names (no '{self._baseName}.nam' file)"
                return self._entries[self._names.index(key)]

    def __iter__(self) :
        yield from self._entries

    def __len__(self) :
        return len(self._entries)

    def __enter__(self) :
        self.open()
        return self
    
    def __exit__(self, *error) :
        self.close()
