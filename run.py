import os
import sys

def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

env_path = os.path.join(sys.prefix, 'Lib', 'site-packages')
packages = os.listdir(env_path)
total_size = 0

for package in packages:
    package_path = os.path.join(env_path, package)
    if os.path.isdir(package_path):
        size = get_size(package_path)
        size_mb = size / (1024 * 1024)
        total_size += size_mb
        print(f"{size_mb:.2f} MB - {package}")

print(f"\nTotal size: {total_size:.2f} MB")
