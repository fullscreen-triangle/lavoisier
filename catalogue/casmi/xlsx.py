"""Minimal stdlib xlsx reader: returns list of rows (list of cell strings)."""
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _colnum(ref):
    n = 0
    for ch in ref:
        if ch.isalpha():
            n = n * 26 + (ord(ch.upper()) - 64)
        else:
            break
    return n - 1


def read_sheet(path, sheet="xl/worksheets/sheet1.xml"):
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in root.findall(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    root = ET.fromstring(z.read(sheet))
    rows = []
    for r in root.iter(NS + "row"):
        cells = {}
        for c in r.findall(NS + "c"):
            ref = c.get("r", "")
            idx = _colnum(ref)
            t = c.get("t")
            v = c.find(NS + "v")
            if t == "s" and v is not None:
                val = shared[int(v.text)]
            elif t == "inlineStr":
                isel = c.find(NS + "is")
                val = "".join(x.text or "" for x in isel.iter(NS + "t")) if isel is not None else ""
            else:
                val = v.text if v is not None else ""
            cells[idx] = val
        if cells:
            w = max(cells) + 1
            rows.append([cells.get(i, "") for i in range(w)])
        else:
            rows.append([])
    return rows


if __name__ == "__main__":
    import sys
    rows = read_sheet(sys.argv[1])
    print("rows:", len(rows))
    for r in rows[: int(sys.argv[2]) if len(sys.argv) > 2 else 12]:
        print(r)
