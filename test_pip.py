from importlib.metadata import version, PackageNotFoundError
from packaging.requirements import Requirement
from packaging.version import Version
from pathlib import Path

req_file = Path("requirements.txt")

if not req_file.exists():
    print("没找到 requirements.txt")
    raise SystemExit(1)

missing = []
mismatch = []
ok = []

for line in req_file.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue

    try:
        req = Requirement(line)
    except Exception as e:
        print(f"无法解析 requirement: {line} -> {e}")
        continue

    name = req.name
    try:
        installed_ver = version(name)
        if req.specifier and Version(installed_ver) not in req.specifier:
            mismatch.append((name, installed_ver, str(req.specifier)))
        else:
            ok.append((name, installed_ver))
    except PackageNotFoundError:
        missing.append((name, str(req.specifier)))

print("\n===== 已安装且版本满足 =====")
for name, ver in ok:
    print(f"{name}=={ver}")

print("\n===== 已安装但版本不满足 =====")
for name, installed, need in mismatch:
    print(f"{name}: 已装 {installed}, 需要 {need}")

print("\n===== 缺失的包 =====")
for name, need in missing:
    print(f"{name} {need}")