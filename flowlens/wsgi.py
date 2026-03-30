import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from flowlens.api import create_app
import yaml

cfg = {}
cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
if cfg_path.exists():
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

app = create_app(cfg)
'''

**1c. Create a `.gitignore`** to keep secrets and junk out of Git:
```
.env
data/
*.db
__pycache__/
*.pyc
.DS_Store
```

**1d. Add a `runtime.txt`** to pin Python version:
```
python-3.11.9
'''