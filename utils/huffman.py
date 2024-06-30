from __future__ import annotations
from typing import Generic, Iterable, Literal, TypeVar

from .bitstream import BitStreamReader, BitStreamWriter

T = TypeVar("T")

class HuffmanNode(Generic[T]) :
    
    def __init__(self, value: T|None = None) -> None:
        self.parent: HuffmanNode|None = None
        self.value = value
        self._weight = 0
        self.childNodes: list[HuffmanNode|None] = [None, None]
    
    def __repr__(self) -> str:
        return f"Huff({self.value}, {self.weight})"
    
    @property
    def weight(self) -> int :
        return self._weight
    
    @weight.setter
    def weight(self, v: int) :
        assert self.childNodes[0] is None
        self._weight = v

    def __getitem__(self, bit: Literal[0, 1]) :
        return self.childNodes[bit]

    def __setitem__(self, bit, child: HuffmanNode) :
        assert 0 <= bit < 2 and self.childNodes[bit] is None and child.parent is None
        self.childNodes[bit] = child
        child.parent = self
        self._weight += child.weight
    
    def setChildren(self, child0: HuffmanNode|None, child1: HuffmanNode|None) :
        if child0 is not None :
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

class HuffmanTable(Generic[T]) :

    def __init__(self, leafs: Iterable[HuffmanNode[T]], invert: bool = False) -> None:
        self.nodes = list(leafs)
        self._nb_leafs = len(self.nodes)
        self._invert = invert
    
    @property
    def size(self) :
        return len(self.nodes)
    
    @property
    def nb_leafs(self) -> int :
        return self._nb_leafs
    
    def __getitem__(self, i: int) -> HuffmanNode:
        return self.nodes[i]
    
    def __iter__(self) :
        yield from self.nodes

     # overload this method in children classes if possible to accelerate compression
    def getNode(self, value: T) -> HuffmanNode[T] :
        for node in self.nodes :
            if node.value == value :
                return node
        raise KeyError(f"No huffman node for value {value}")
    
    def buildTree(self, max_table_size: int) -> None :
        
        total_weight = 0
        for i in range(0, self.nb_leafs) :
            total_weight += self.nodes[i].weight
        
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
            assert child0 is not None, \
                "Could not build huffman tree"
            parentNode = HuffmanNode()
            if self._invert :
                child0, child1 = child1, child0
            parentNode.setChildren(child0, child1)
            self.nodes.append(parentNode)
            if parentNode.weight >= total_weight :
                break
    
    def decodeSequence(self, src: BitStreamReader) -> T :
        node = self.nodes[self.size-1]
        while node.value is None :
            node = node[src.readBit()]
            assert node is not None
        return node.value
    
    def encodeValue(self, dest: BitStreamWriter, value: T) -> None :
        node = self.getNode(value)
        bin, size = node.encode()
        dest.write(size, bin)
    

class IntHuffmanTable(HuffmanTable[int]) :

    def __init__(self, leaf_values: Iterable[int], invert=False) -> None:
        super().__init__(map(HuffmanNode, leaf_values), invert)

class ByteHuffmanTable(IntHuffmanTable) :

    def __init__(self, invert=False) -> None:
        super().__init__(range(0, 256), invert)
    
    def getNode(self, value: int) -> HuffmanNode :
        return self.nodes[value]