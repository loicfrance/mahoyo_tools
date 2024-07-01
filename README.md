# mahoyo-tools
extract and reinject files into Witch of the Holy Night (Steam)


Requires python >= 3.10, numpy, Pillow, zlib

## Open `.hfa` Archive

`.hfa` archive files contents are accessed through `HfaArchive` objects.

```python
import os
from mahoyo_tools import hfa
GAME_DIR = "path/to/steamapps/common/WITCH ON THE HOLY NIGHT"
hfa_name = "data00000"
hfa_path = os.path.join(game_dir, f"{hfa_name}.hfa")

archive = hfa.HfaArchive(hfa_path)
```

An optional `mode` parameter can be set to `'rw'` in the `HfaArchive`
constructor to automatically replace the original file content when the archive
is closed. Otherwise, the original file will stay untouched unless specifically
written to.

```python
archive = hfa.HfaArchive(hfa_path, 'rw')
```

To access the content, open the archive using either the `open()` method or
the `with` syntax.
Once all accesses to the archive content is complete, use the `close()` method
to close the file or exit the `with` block.
If the `mode` parameter in the constructor was set to `'rw'`, then the original
file content is replaced when closing the file.

## Access archive entries

An opened archive maps entry names or indices to `HfaArchiveEntry` objects
that hold entry info and are used to access entry data.

```python
entry = archive[entry_name]
# or
entry = archive[index]
```
Enties can be used as files to access the entry data, through one of the
following methods :
  - `read()`, `write()`, `seek()` and `tell()` that reproduce the behaviour of
    python's `io.BufferedRandom` objects;
  - `data` property to get and set all the data bytes of the entry;
  - `to_file()` and `from_file()` to load in / from a file all the entry data
    bytes;
  - `extract()` and `inject()` that decompress the content of the entry to a
    file, or compress the content of the file to the entry bytes (c.f. below).

## Extract / Inject content to an entry

Contents of an archive entry can be extracte or injected using the `extract()`
and `inject()` methods.

When possible, the `extract()` method decompresses the content of the archive
entry to the destination file passed in parameter.
The `inject()` method compresses the content of a file back to the archive
entry.

Some limitations apply, depending on the file format :
 - `mzp` images can be extracted but not injected yet;
 - `ctd` scripts are decompressed, but are reinjected as-is, as the
    _LenZuCompressor_ compression algorithm has not yet been efficiently
    re-written (this should not impact the game);
 - `ccit` font files are not transformed to an other format. use the `Font`
    class (described below) to modify the font files;
 - `chs` files cannot be decompiled yet, and are therefore extracted and
    re-injected as-is;

## Modifying the fonts

The fonts in the game are separated in multiple files and formats :
 - `mzp` images where the characters are drawn
 - `ccit` files that list all the characters and their position in the images

`mzp` font images can be generated properly using the `font.svg` file included
in the repository. A guide to generate the images using **Inkscape** is included in
it.

`ccit` files can be generated using the `ccit.Font` class, like in the example
below :

```python
# { character: width }
font = Font({
    ' ':26, '!':21, '"':27, '#':45, '$':45, '%':65, '&':52, "'":17, '(':27, ')':27, '*':33, '+':45, ',':21, '-':31,
    '.':21, '/':37, '0':43, '1':30, '2':43, '3':44, '4':44, '5':44, '6':43, '7':43, '8':43, '9':44, ':':21, ';':21,
    '<':45, '=':45, '>':45, '?':41, '@':59, 'A':55, 'B':49, 'C':54, 'D':51, 'E':45, 'F':43, 'G':55, 'H':49, 'I':19,
    'J':40, 'K':51, 'L':43, 'M':59, 'N':49, 'O':57, 'P':47, 'Q':57, 'R':50, 'S':49, 'T':49, 'U':49, 'V':53, 'W':67,
    'X':53, 'Y':55, 'Z':49, '[':25,'\\':37, ']':25, '^':45, '_':45, '`':26, 'a':44, 'b':45, 'c':43, 'd':45, 'e':44,
    'f':32, 'g':43, 'h':41, 'i':19, 'j':24, 'k':42, 'l':19, 'm':59, 'n':41, 'o':45, 'p':45, 'q':45, 'r':31, 's':41,
    't':31, 'u':41, 'v':43, 'w':59, 'x':45, 'y':43, 'z':41, '{':30, '|':17, '}':30, '~':43, '±':45, '·':21, '×':43,

    'à':44, 'è':44, 'ì':19, 'ò':45, 'ù':41, 'ỳ':43, 'á':44, 'é':44, 'í':19, 'ó':45, 'ú':41, 'ý':43, 'æ':58, 'œ':60,
    'â':44, 'ê':44, 'î':24, 'ô':45, 'û':41, 'ŷ':43, 'ä':44, 'ë':44, 'ï':24, 'ö':45, 'ü':41, 'ÿ':43, 'ç':43, 'ß':35,
    'ā':44, 'ē':44, 'ī':24, 'ō':45, 'ū':41, 'å':44, 'ñ':41, '❶':26, '❷':26, '❸':26, '❹':26, '❺':26, '❻':26, '❼':26,
    'À':55, 'È':45, 'Ì':19, 'Ò':57, 'Ù':49, 'Ỳ':55, 'Á':55, 'É':45, 'Í':19, 'Ó':57, 'Ú':49, 'Ý':55, 'Æ':65, 'Œ':71,
    'Â':55, 'Ê':45, 'Î':24, 'Ô':57, 'Û':49, 'Ŷ':55, 'Ä':55, 'Ë':45, 'Ï':24, 'Ö':57, 'Ü':49, 'Ÿ':55, 'Ç':55, '❽':26,
    'Ā':55, 'Ē':45, 'Ī':24, 'Ō':57, 'Ū':49, 'Å':55, 'Ñ':49, '❾':26, '❿':26, '⓫':26, '⓬':26, '⓭':26, '⓮':26, '⓯':26,
    '÷':45, '¡':21, '¿':41, '「':73, '」':73, '『':73, '』':73, '«': 43, '»': 43, 'ー': 73, '²': 28
})

with HfaArchive(f"{path_to_game}/data00100.hfa", 'rw') as archive :
    archive["Font010000.ccit"].inject(font.to_ccit())
```