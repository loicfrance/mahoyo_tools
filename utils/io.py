
from io import BytesIO
import os
from typing import Protocol, TYPE_CHECKING, overload

if TYPE_CHECKING :
    from _typeshed import ReadableBuffer


class BytesIOBase(Protocol) :
    def tell(self) -> int : ...
    def seek(self, target: int, whence: int = 0, /) -> int : ...

class BytesReader(BytesIOBase, Protocol) :
    def read(self, size: int | None = ..., /) -> bytes : ...

class BytesWriter(BytesIOBase, Protocol) :
    def write(self, data: 'ReadableBuffer', /) -> int : ...
    def truncate(self, size: int | None = ..., /) -> int: ...

class BytesRW(BytesReader, BytesWriter, Protocol) :
    pass

@overload
def save(src: bytes | BytesReader, dest: str | BytesWriter) -> None : ...
@overload
def save(src: bytes | BytesReader, dest: None = None) -> BytesIO : ...
def save(src: bytes | BytesReader, dest: str | BytesWriter | None = None) :
    if not isinstance(src, (bytes, memoryview, bytearray)) :
        src = src.read()
    match dest :
        case None :
            return BytesIO(src)
        case str() :
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            assert not os.path.isdir(os.path.abspath(dest))
            file = open(dest, "rb+" if os.path.exists(dest) else "wb+")
            file.write(src)
            file.close()
        case _ :
            dest.write(src)
