"""pytest 配置：把项目根加入 sys.path，避免每个 test 文件重复 sys.path.insert"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
