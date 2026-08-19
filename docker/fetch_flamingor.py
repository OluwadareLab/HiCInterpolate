#!/usr/bin/env python3
import json
import os
import urllib.request

TREE_URL = "https://api.github.com/repos/wangjr03/FLAMINGO/git/trees/main?recursive=1"
CDN = "https://cdn.jsdelivr.net/gh/wangjr03/FLAMINGO@main/"
ROOT = "/tmp/FLAMINGO"


def main():
    with urllib.request.urlopen(TREE_URL, timeout=120) as resp:
        tree = json.load(resp)
    n = 0
    for item in tree["tree"]:
        if item["type"] != "blob" or not item["path"].startswith("FLAMINGOr/"):
            continue
        dest = os.path.join(ROOT, item["path"])
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        urllib.request.urlretrieve(CDN + item["path"], dest)
        n += 1
    print("downloaded", n, "FLAMINGOr files")
    if n < 10:
        raise SystemExit("FLAMINGOr download incomplete")


if __name__ == "__main__":
    main()
