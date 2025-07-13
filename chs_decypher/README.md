# CHS scripts decyphering

This directory contains markdown files and (yet-to-be created) python scripts
to help decypher the `chs` (Compiled Hunex Script) format used by the game
for its scripts, and _hopefully_ decompile it
into its `.hss` (Hunex Script Source ?) equivalent file.

## CHS file structure

CHS files are made of three sections: the header, the instructions,
and the arrays.

Strings and raw values can are addressed with their position in the file,
minus the size of the first subsection of the header (`30h` = 48)

### 1. Header

The header of a `chs` is a has the following structure:

|   Byte number   |               Description              |
|-----------------|----------------------------------------|
| `0000h`-`001Ch` | Format: `HunexCompiledScriptVer1.00\0` |
| `001Bh`         | Offset to end of header (`15h` = 21)   |
| `001Ch`-`002Fh` | Empty bytes                            |

### 2. Instructions

#### 2.1. Stack machine

the HuneX engine uses a stack machine, where values are pushed onto the stack,
and popped by instructions to perfom all the actions.

All values and operations are encoded onto 4-byte dwords in little endian.
String values are accessed by their address in the file, with a 48 (=`30h`)
offset correspondong to the header size.

### 2.2. Base opcodes

Hunex scripts uses the same set of basic instructions as buriko scripts.
Its opcodes are listed in [this github repository][opcodes_bgi].
When an operation requires arguments, the engine uses the next dwords for it.
Operations can also pop values from the stack.

<!--region opcodes -->
<details><summary> Base opcodes list </summary>

| Code | Name                  | Arguments
|-----:|:----------------------|:---------
|`00h` | PushDword             | Dword
|`01h` | PushOffset            | Offset
|`02h` | PushPCOffset          | Dword
|`03h` | PushString            | String
|`08h` | ReadMemory            | Dword
|`09h` | WriteMemory           | Dword
|`0Ah` | WriteMemoryArgs       | Dword
|`10h` | PushPC                |
|`11h` | PopPC                 |
|`18h` | Jmp                   |
|`19h` | Jmp if Condition      | Dword
|`1Ah` | Call                  |
|`1Bh` | Return                |
|`1Eh` | ExceptionHandler      |
|`1Fh` | UnregExceptionHandler |
|`20h` | Add                   |
|`21h` | Sub                   |
|`22h` | Imul                  |
|`23h` | Idlv                  |
|`24h` | Mod                   |
|`25h` | And                   |
|`26h` | Or                    |
|`27h` | Xor                   |
|`28h` | Not                   |
|`29h` | Shl                   |
|`2Ah` | Shr                   |
|`2Bh` | Sar                   |
|`30h` | SetE                  |
|`31h` | SetNE                 |
|`32h` | SetLE                 |
|`33h` | SetGE                 |
|`34h` | SetL                  |
|`35h` | SetG                  |
|`38h` | AndBool               |
|`39h` | OrBool                |
|`3Ah` | ZeroBool              |
|`3Fh` | GetArgs               | Dword
|`60h` | CopyMemory            |
|`61h` | ZeroMemory            |
|`62h` | FillMemory            |
|`63h` | CmpMemory             |
|`66h` | CmpString             |
|`68h` | Strlen                |
|`69h` | CmpString2            |
|`6Ah` | Strcpy                |
|`6Bh` | ErrorMsg              |
|`6Ch` | IsPunctuation         |
|`6Dh` | StringUpper           |
|`6Fh` | StringFormat          |
|`74h` | Change                |
|`78h` | YesNoBox              |
|`79h` | FatalBox              |
|`7Ch` | ModelBox              |
|`7Eh` | CopyToClipBoard       |
|`7Fh` | Debug                 | String, Dword
|`80h` | CallFunction1         |
|`81h` | CallFunction2         |
|`90h` | CallFunction3         |
|`91h` | CallFunction4         |
|`92h` | CallFunction5         |
|`a0h` | CallFunction6         |
|`b0h` | CallFunction7         |
|`c0h` | CallFunction8         |

Codes `80h` to `c0h` are syscalls.
</details>

<!--endregion-->

### 2.3. Hunex commands

On top of this opcodes list, Hunex script also implement its commands using
operation codes.

<!--region commands -->
<details>
<summary>Hunex Known commands list</summary>

_WIP_

|  Code | Name      | Description
|------:|:----------|:-----------
|`00E9h`| _WKST     |
|`0110h`| _WTTM     | Wait timer
|`0128h`| _RDST     |
|`012Ah`| _PGST     | Set page number
|`0140h`| _ZM01xxx  | Write text
|`0142h`| _wrel     | Write empty line (arbitrary name)
|`0152h`| _CLO4     |
|`0180h`| _MPLY     |
|`0184h`| _MSTP     |
|`0185h`| _MFAD     |
|`0186h`| _MVOL     |
|`0190h`| _SEPL     | Play sound effect
|`0195h`| _SEFD     |
|`0196h`| _SEVL     |
|`01A0h`| _VPLY     | Play voice
|`0200h`| _FADS     |
|`0240h`| _STCH     |
|`0241h`| _STCC     |
|`0242h`| _SCH2     |
|`0243h`| _STCL     |
|`0245h`| _STGS     |
|`0246h`| _STZ4     |
|`0247h`| _STRT     |
|`0251h`| _STCP     |
|`0252h`| _STBR     |
|`0253h`| _STEC     |
|`0254h`| _SNX3     |
|`0256h`| _SNX4     |
|`0257h`| _STNS     |
|`025Ah`| _SACL     |
|`0270h`| _STQK     |
|`0272h`| _STQS     |
|`0278h`| _STMA     |
|`027Fh`| _STCF     |

`SE`-prefixed commands affect sound effects,
`ST`-prefixed commands affect sprites.

</details>

<!--endregion-->

### 2.4. Other information

For debug purpose, compiled Hunex script files link the original files and
line numbers using the `7Fh` opcode. This operation is called whenever the line
and/or file changes in the original source file before executing an instruction,
in order to display the proper message if an error occurs.

### 3. Arrays

Strings and arrays are located at the end of the file, and addressed by their
position in the file, minus the header size.

Strings always end with byte `00h` (character `'\0'`).


[opcodes_bgi]: https://github.com/xmoezzz/BGITool/blob/7b2ce6dd5febe3b806e07b705e3c28e5f541825a/BGIDisasm/BGIDisasm/Instruction.cpp#L4