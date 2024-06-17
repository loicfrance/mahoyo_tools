# mahoyo-tools
extract and reinject files into Witch of the Holy Night (Steam)


Requires python >= 3.10, numpy, Pillow, zlib

## Open `.hfa` Archive
```python
import os
from mahoyo_tools import hfa

GAME_DIR = "path/to/steamapps/common/WITCH ON THE HOLY NIGHT"

hfa_name = "data00000"

hfa_path = os.path.join(game_dir, f"{hfa_name}.hfa")

archive = hfa.HfaArchive(hfa_path)
archive.open()
# ... do stuff
archive.close()

# or
with hfa.HfaArchive(hfa_path) as archive :
    # ... do stuff
```




