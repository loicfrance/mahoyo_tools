"""
Credits to mangetsu (https://github.com/rschlaikjer/mangetsu) for details on
archive files structure.
"""

################################################################################
#region                        imports, constants
################################################################################
from __future__ import annotations
import os
import struct
from typing import BinaryIO, Iterable, Literal

NAM_ENTRY_LEN = 32
HED_ENTRY_LEN = 8
MRG_SECTOR_SIZE = 0x800

#endregion #####################################################################
#region                                HED
################################################################################

def readHED(path: str) :
    entries: list[tuple[int, int, int]] = []
    assert os.path.exists(path), f"file '{path}' does not exist"
    with open(path, 'rb') as file :
        size = file.seek(0, 2)
        assert size % HED_ENTRY_LEN == 0, \
            f"wrong size for hed file {path}: {size}"
        
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

#endregion #####################################################################
#region                                NAM
################################################################################

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
            name = buffer.decode('utf-8').strip('\r\n')
            fileNames.append(name)
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
#region                                MRG
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
    
    @property
    def compressed(self) :
        return self._size_decomp_sectors == self._size_comp_sectors
    
    @property
    def data(self) -> bytes :
        file = self._mrg._mrgFile
        assert file is not None, "mrg file is closed"
        pos = file.tell()
        file.seek(self._offset_sectors * MRG_SECTOR_SIZE)
        buffer = file.read(self._size_comp_sectors * MRG_SECTOR_SIZE)
        file.seek(pos)
        return buffer

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
