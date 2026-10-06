import argparse
import csv
import html
import json
from pathlib import Path

CRITERIA = [
    ("style", "Style: how much it looks like a Monet painting (5 = very Monet)"),
    ("content", "Content: how well the scene of the photo is kept (5 = fully kept)"),
    ("artifacts", "Artifacts: visual defects (5 = none, 1 = severe)"),
]


def load_sheet(path):
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        header = list(rows[0].keys()) if rows else []
    return header, rows


def build(sheet, images, inputs, out, title):
    header, rows = load_sheet(sheet)
    blind = "blind_id" in header
    items = []
    for row in rows:
        if blind:
            key = row["blind_id"]
            img = Path(images) / f"{key}.jpg"
            inp = Path(inputs) / row["input_file"] if inputs else None
            prefill = {c: row.get(f"{c}_1to5", "") for c, _ in CRITERIA}
        else:
            key = row["sample"]
            img = Path(images) / f"{key}.png"
            if not img.exists():
                img = Path(images) / f"{key}.jpg"
            inp = None
            prefill = {c: row.get(c, "") for c, _ in CRITERIA}
        items.append({
            "key": key,
            "img": img.resolve().as_uri(),
            "inp": inp.resolve().as_uri() if inp and inp.exists() else None,
            "prefill": prefill,
            "extra": row.get("input_file", ""),
        })
    cards = []
    for i, it in enumerate(items, 1):
        pics = ""
        if it["inp"]:
            pics += f'<figure><img src="{it["inp"]}"><figcaption>input photo</figcaption></figure>'
            pics += f'<figure><img src="{it["img"]}"><figcaption>translation</figcaption></figure>'
        else:
            pics += f'<figure><img src="{it["img"]}" class="wide"><figcaption>left: input photo, right: translation</figcaption></figure>'
        radios = ""
        for c, label in CRITERIA:
            opts = "".join(
                f'<label><input type="radio" name="{it["key"]}__{c}" value="{v}" {"checked" if str(it["prefill"].get(c, "")).strip() == str(v) else ""}> {v}</label>'
                for v in range(1, 6)
            )
            radios += f'<div class="crit"><span>{html.escape(label)}</span><div class="opts">{opts}</div></div>'
        cards.append(f'<section class="card" id="card-{i}"><h3>{i} of {len(items)}: {html.escape(it["key"])}</h3><div class="pics">{pics}</div>{radios}</section>')
    keys = json.dumps([it["key"] for it in items])
    crit_keys = json.dumps([c for c, _ in CRITERIA])
    header_json = json.dumps(header)
    extra_json = json.dumps({it["key"]: it["extra"] for it in items})
    filename = Path(sheet).name
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f4f4;color:#222}}
header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ccc;padding:10px 20px;display:flex;gap:20px;align-items:center;z-index:2}}
header h1{{font-size:18px;margin:0;flex:1}}
button{{font-size:15px;padding:8px 14px;cursor:pointer}}
.card{{background:#fff;margin:16px auto;max-width:1100px;padding:14px 20px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,.15)}}
.pics{{display:flex;gap:16px;flex-wrap:wrap}}
figure{{margin:0}} figure img{{max-height:320px;border:1px solid #ddd}} figure img.wide{{max-width:100%;max-height:360px}}
figcaption{{font-size:12px;color:#666}}
.crit{{display:flex;align-items:center;gap:16px;margin:8px 0}} .crit span{{flex:1;font-size:14px}}
.opts label{{margin-right:10px;font-size:15px;cursor:pointer}}
.done{{border-left:6px solid #2a9d4a}}
#status{{font-size:14px;color:#444}}
</style></head><body>
<header><h1>{html.escape(title)}</h1><span id="status"></span><button onclick="download()">Download CSV</button></header>
<p style="max-width:1100px;margin:16px auto;font-size:14px">Score each translation from 1 to 5 on the three criteria. Judge only the translation, not the input photo. Your choices are saved in this browser as you go; when the counter shows all rated, click Download CSV and the filled sheet <b>{html.escape(filename)}</b> is saved to your Downloads folder.</p>
{''.join(cards)}
<script>
const KEYS={keys}, CRIT={crit_keys}, HEADER={header_json}, EXTRA={extra_json}, STORE="audit-"+{json.dumps(filename)};
const saved=JSON.parse(localStorage.getItem(STORE)||"{{}}");
for(const k in saved){{const el=document.querySelector(`input[name="${{k}}"][value="${{saved[k]}}"]`); if(el) el.checked=true;}}
function val(k,c){{const el=document.querySelector(`input[name="${{k}}__${{c}}"]:checked`); return el?el.value:"";}}
function update(){{let n=0; KEYS.forEach((k,i)=>{{const ok=CRIT.every(c=>val(k,c)); if(ok) n++; document.getElementById("card-"+(i+1)).classList.toggle("done",ok);}}); document.getElementById("status").textContent=n+" of "+KEYS.length+" rated"; const s={{}}; document.querySelectorAll("input[type=radio]:checked").forEach(e=>s[e.name]=e.value); localStorage.setItem(STORE,JSON.stringify(s));}}
document.querySelectorAll("input[type=radio]").forEach(e=>e.addEventListener("change",update)); update();
function download(){{
  const blind=HEADER.includes("blind_id");
  let lines=[HEADER.join(",")];
  KEYS.forEach(k=>{{const row=HEADER.map(h=>{{ if(h==="blind_id"||h==="sample") return k; if(h==="input_file") return EXTRA[k]; const c=h.replace("_1to5",""); return CRIT.includes(c)?val(k,c):""; }}); lines.push(row.join(","));}});
  const blob=new Blob([lines.join("\\n")+"\\n"],{{type:"text/csv"}}); const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download={json.dumps(filename)}; a.click();
}}
</script></body></html>"""
    Path(out).write_text(page, encoding="utf-8")
    print(f"wrote {out} with {len(items)} samples; download produces {filename}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--inputs", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Human audit")
    args = ap.parse_args()
    build(args.sheet, args.images, args.inputs, args.out, args.title)


if __name__ == "__main__":
    main()
