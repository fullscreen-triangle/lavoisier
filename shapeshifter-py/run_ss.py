import sys, json
from shapeshifter import run

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m shapeshifter <program.ss> [out_dir]")
        sys.exit(2)
    src_path = sys.argv[1]
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else "results"
    name = src_path.split("/")[-1].split(chr(92))[-1].replace(".ss", "")
    with open(src_path, encoding="utf-8") as fh:
        source = fh.read()
    payload = run(source, out_dir=out_dir, program_name=name)
    for line in payload["compile"]["terminal"]:
        print(line["text"])
    if payload["execute"]:
        for line in payload["execute"]["terminal"]:
            print(line["text"])
    print(f"-> results written to {out_dir}/")
