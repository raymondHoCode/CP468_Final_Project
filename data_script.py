import os
import re
import urllib.request

WIKIAUTO = "https://raw.githubusercontent.com/chaojiang06/wiki-auto/master/wiki-auto/ACL2020"
ASSET = "https://raw.githubusercontent.com/facebookresearch/asset/main/dataset"

UNESCAPE = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
            "-LCB-": "{", "-RCB-": "}", "``": '"', "''": '"'}


def get(url, name):
    path = os.path.join("data", "raw", name)
    if not os.path.exists(path):
        print(f"  {name}")
        urllib.request.urlretrieve(url, path)
    with open(path, encoding="utf-8") as f:
        lines = []
        for line in f:
            for a, b in UNESCAPE.items():
                line = line.replace(a, b)
            lines.append(re.sub(r"\s+", " ", line).strip())
    return lines


def save(name, lines):
    with open(os.path.join("data", name), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


os.makedirs(os.path.join("data", "raw"), exist_ok=True)

print("WikiAuto (training)")
src = get(f"{WIKIAUTO}/train.src", "wa.src")
dst = get(f"{WIKIAUTO}/train.dst", "wa.dst")

kept = []
dropped = {"identical": 0, "too short": 0, "too long": 0, "target longer": 0}
for s, d in zip(src, dst):
    sw, dw = s.split(), d.split()
    if not sw or not dw or len(sw) < 3 or len(dw) < 3:
        dropped["too short"] += 1
    elif s == d:
        dropped["identical"] += 1
    elif len(sw) > 80 or len(dw) > 80:
        dropped["too long"] += 1
    elif len(dw) > 1.5 * len(sw):
        dropped["target longer"] += 1
    else:
        kept.append((s, d))

save("train.src", [s for s, _ in kept])
save("train.dst", [d for _, d in kept])
print(f"  {len(kept):,} pairs kept of {len(src):,}")
print("  dropped: " + ", ".join(f"{k} {v:,}" for k, v in dropped.items()))

for split, out in [("valid", "valid"), ("test", "test_asset")]:
    print(f"ASSET ({split})")
    sources = get(f"{ASSET}/asset.{split}.orig", f"as.{split}.orig")
    save(f"{out}.src", sources)
    for i in range(10):
        refs = get(f"{ASSET}/asset.{split}.simp.{i}", f"as.{split}.{i}")
        assert len(refs) == len(sources)
        save(f"{out}.ref.{i}", refs)
    print(f"  {len(sources):,} sentences x 10 references")