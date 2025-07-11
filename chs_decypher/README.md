# CHS scripts decyphering

This directory contains markdown files and python scripts to help decypher the
`chs` (Compiled Hunex Script) format used by the game for its scripts,
and _hopefully_ decompile it into its `.hss` (Hunex Script Source ?) equivalent.

## CHS file structure

CHS files are made of four sections: the header, the instructions, the strings and the raw values.

### Header

The header of a `chs` has 2 subsection:

1. Format

    |   Byte number   |               Description              |
    |-----------------|----------------------------------------|
    | `0000h`-`001Ch` | Format: `HunexCompiledScriptVer1.00\0` |
    | `001Bh`         | offset to next section (`15h` = 21)    |

2. Addesses
    bytes `0030h` to `0073h` are grouped into dwords in little-endian.
    The first two seem to always be empty (`0000h 0000h`).

    The following ones are split into two groups using the same pattern :
    |  0  |  1  |  2  |  3  |  4  |  4  |     6     |
    |:---:|:---:|:---:|:---:|:---:|:---:|:---------:|
    |`7Fh`|`01h`|`44h`|`19h`|`01h`|`03h`|`XXXXXXXXh`|

    The dword 6 is an address. The first pattern occurence
    specifies the address of the string `"START"`, and the second occurrence
    specifies the address of the beginning of raw values in the file (or the end
    of the strings section)

### Instructions

Instructions start at address `0070h` with the dword `000000F9h` in
little-endian.

Instructions are made of dword values in little endian, and have the following
structure :

It starts with the value `F7h`, then list all of its parameters,
and finally specifies the command name.

The parameters are coded using opcodes and values when necessary. The opcodes
used by hunex scripts are the same as the one used by buriko scripts, which are
listed in [this github repository][opcodes_bgi].

<!--region opcodes -->
<details><summary> Opcodes list </summary>

| Code | Name                   | Arguments     |
|-----:|:-----------------------|:--------------|
|`00h` | PushDword              | Dword         |
|`01h` | PushOffset             | Offset        |
|`02h` | PushPCOffset           | Dword         |
|`03h` | PushString             | String        |
|`08h` | ReadMemory             | Dword         |
|`09h` | WriteMemory            | Dword         |
|`0Ah` | WriteMemoryArgs        | Dword         |
|`10h` | PushPC                 |               |
|`11h` | PopPC                  |               |
|`18h` | Jmp                    |               |
|`19h` | JC                     | Dword         |
|`1Ah` | Call                   |               |
|`1Bh` | Return                 |               |
|`1Eh` | ExceptionHandler       |               |
|`1Fh` | UnregExceptionHandler  |               |
|`20h` | Add                    |               |
|`21h` | Sub                    |               |
|`22h` | Imul                   |               |
|`23h` | Idlv                   |               |
|`24h` | Mod                    |               |
|`25h` | And                    |               |
|`26h` | Or                     |               |
|`27h` | Xor                    |               |
|`28h` | Not                    |               |
|`29h` | Shl                    |               |
|`2Ah` | Shr                    |               |
|`2Bh` | Sar                    |               |
|`30h` | SetE                   |               |
|`31h` | SetNE                  |               |
|`32h` | SetLE                  |               |
|`33h` | SetGE                  |               |
|`34h` | SetL                   |               |
|`35h` | SetG                   |               |
|`38h` | AndBool                |               |
|`39h` | OrBool                 |               |
|`3Ah` | ZeroBool               |               |
|`3Fh` | GetArgs                | Dword         |
|`60h` | CopyMemory             |               |
|`61h` | ZeroMemory             |               |
|`62h` | FillMemory             |               |
|`63h` | CmpMemory              |               |
|`66h` | CmpString              |               |
|`68h` | Strlen                 |               |
|`69h` | CmpString2             |               |
|`6Ah` | Strcpy                 |               |
|`6Bh` | ErrorMsg               |               |
|`6Ch` | IsPunctuation          |               |
|`6Dh` | StringUpper            |               |
|`6Fh` | StringFormat           |               |
|`74h` | Change                 |               |
|`78h` | YesNoBox               |               |
|`79h` | FatalBox               |               |
|`7Ch` | ModelBox               |               |
|`7Eh` | CopyToClipBoard        |               |
|`7Fh` | Debug                  | String, Dword |
|`80h` | CallFunction1          |               |
|`81h` | CallFunction2          |               |
|`90h` | CallFunction3          |               |
|`91h` | CallFunction4          |               |
|`92h` | CallFunction5          |               |
|`a0h` | CallFunction6          |               |
|`b0h` | CallFunction7          |               |
|`c0h` | CallFunction8          |               |

Codes `80h` to `c0h` are syscalls.
</details>

<!--endregion-->

The known command codes, equivalent `hss` names and arguments are listed below.
They are deducted by comparing the compiled `chs` from the steam version with
`hss` files from the switch version.

<!--region commands -->
<details>
<summary>Commands list</summary>

_WIP_

|  Code | Name      | Arguments | Description |
|------:|:----------|:----------|:------------|
|`00E9h`| _WKST     |
|`0110h`| _WTTM     |
|`0128h`| _RDST     |
|`012Ah`| _PGST     |
|`0140h`| _ZM01xxx  |
|`0142h`| _wrel     |
|`0152h`| _CLO4     |
|`0180h`| _MPLY     |
|`0184h`| _MSTP     |
|`0185h`| _MFAD     |
|`0186h`| _MVOL     |
|`0190h`| _SEPL     |
|`0195h`| _SEFD     |
|`0196h`| _SEVL     |
|`01A0h`| _VPLY     |
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

</details>

<!--endregion-->

### Strings

_TODO_

### Raw values

_TODO_

[opcodes_bgi]: https://github.com/xmoezzz/BGITool/blob/7b2ce6dd5febe3b806e07b705e3c28e5f541825a/BGIDisasm/BGIDisasm/Instruction.cpp#L4