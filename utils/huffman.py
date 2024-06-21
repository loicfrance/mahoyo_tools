from __future__ import annotations
from abc import ABC, abstractmethod
from io import BufferedReader, BufferedWriter, BytesIO
from typing import Any, Generic, Iterable, Literal, TypeVar, cast, overload


T = TypeVar("T")
class HuffmanNode(Generic[T]) :
    
    def __init__(self, value: T|None = None) -> None:
        self.parent: HuffmanNode|None = None
        self.value = value
        self._weight = 0
        self.childNodes: list[HuffmanNode|None, HuffmanNode|None] = [None, None]
    
    def __repr__(self) -> str:
        return f"Huff({self.value}, {self.weight})"
    
    @property
    def weight(self) -> int :
        return self._weight
    
    @weight.setter
    def weight(self, v: int) :
        assert self[0] is None
        self._weight = v

    def __getitem__(self, bit: Literal[0, 1]) :
        return self.childNodes[bit]

    def __setitem__(self, bit, child: HuffmanNode) :
        assert 0 <= bit < 2 and self[bit] is None and child.parent is None
        if bit == 1 :
            assert self[0] is not None
        self.childNodes[bit] = child
        child.parent = self
        self._weight += child.weight
    
    def setChildren(self, child0: HuffmanNode, child1: HuffmanNode|None) :
        self[0] = child0
        if child1 is not None :
            self[1] = child1
    
    def encode(self) -> tuple[int, int] :
        if self.parent is None :
            return (0,0)
        value, size = self.parent.encode()
        if self.parent[0] is self :
            return (value, size + 1)
        else :
            return (value | (1 << size), size + 1)

class HuffmanTable(ABC) :

    def __init__(self) -> None:
        self.nodes = list(self._createInitNodes())
    
    @property
    def size(self) :
        return len(self.nodes)
    
    @property
    @abstractmethod
    def nb_leafs(self) -> int : ...

    @abstractmethod
    def _createInitNodes(self) -> Iterable[HuffmanNode]: ...

    @abstractmethod
    def _writeNode(self, dest: BufferedWriter, nodeValue: Any) -> None : ...

    @abstractmethod
    def _readNode(self, src: BufferedReader) -> HuffmanNode : ...
    
    def __getitem__(self, i: int) -> HuffmanNode:
        return self.nodes[i]
    
    def __iter__(self) :
        yield from self.nodes
    
    def createHierarchy(self, max_table_size: int) :
        
        total_weight = 0
        for i in range(0, self.nb_leafs) :
            total_weight += self[i].weight
        
        for i in range(self.nb_leafs, max_table_size) :
            child0: HuffmanNode|None = None
            child1: HuffmanNode|None = None

            for j, node in enumerate(self) :
                if j >= i : break
                if node.weight == 0 or \
                   node.parent is not None :
                   continue

                if child0 is None or node.weight < child0.weight :
                    child1 = child0
                    child0 = node
                elif child1 is None or node.weight < child1.weight :
                    child1 = node
            parentNode = HuffmanNode()
            parentNode.setChildren(child0, child1)
            self.nodes.append(parentNode)
            if parentNode.weight >= total_weight :
                break
    
    def decompress(self, src: BufferedReader, decomp_size: int) :
        output = BytesIO()
        output.truncate(decomp_size)
        output.seek(0)
        shift = 8
        byte = 0
        for _ in range(0, decomp_size) :
            node = self[self.size-1]
            while node.value is None :
                if shift == 8 :
                    byte = src.read(1)[0]
                    node = node[byte & 1]
                    shift = 1
                else :
                    node = node[(byte >> shift) & 1]
                    shift += 1
            self._writeNode(output, node)
        output.seek(0) 
        return output
    
    def compress(self, src: BufferedReader, decomp_size: int) :
        output = BytesIO()
        shift = 0
        byte = 0
        while src.tell() < decomp_size :
            node = self._readNode(src)
            binary, size = node.encode()
            byte |= binary << shift
            shift += size
            while shift >= 8 :
                output.write((byte & 0xFF).to_bytes(1))
                byte >>= 8
                shift -= 8
        if shift > 0 :
            output.write(byte.to_bytes(1))
        output.seek(0)
        return output

class ByteHuffmanTable(HuffmanTable) :

    @property
    def nb_leafs(self) -> int :
        return 256
    
    def _writeNode(self, dest: BufferedWriter, node: HuffmanNode) -> None :
        dest.write(cast(int, node.value).to_bytes(1))
    
    def _readNode(self, src: BufferedReader) -> HuffmanNode :
        byte = src.read(1)[0]
        return self[byte]
    
    def _createInitNodes(self) -> Iterable[HuffmanNode]:
        return map(HuffmanNode, range(0, 256))

