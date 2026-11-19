"""Command-line wrapper: validate every GIF in a directory."""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relic_nav.gifcheck import verify


def main():
    paths = sorted(glob.glob(os.path.join(sys.argv[1], "*.gif")))
    ok = True
    for path in paths:
        try:
            w, h, frames = verify(path)
            print(f"OK   {os.path.basename(path):42s} {w}x{h} frames={frames}")
        except Exception as exc:
            ok = False
            print(f"FAIL {os.path.basename(path):42s} {type(exc).__name__}: {exc}")
    print("ALL VALID:", ok)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
