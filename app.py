# -*- coding: utf-8 -*-
import os, io, re, time, zipfile, traceback
from datetime import datetime, timezone, timedelta
from typing import List
from urllib.parse import quote

import pandas as pd
import numpy as np
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, PlainTextResponse

# ====== 설정 ======
HEADER_ROW_IDX = 2
TARGET_POLICY = "개인 맞춤 광고 정책 내 건강 관련 콘텐츠 (제한됨)"
TARGET_SIZES  = ["1x1", "4x5", "9x16", "1920x1080"]

NOTION_TOKEN   = os.getenv("NOTION_TOKEN", "")
NOTION_DB_ID   = os.getenv("NOTION_DB_ID", "")
NOTION_VERSION = "2022-06-28"


# ====== 유틸 ======
def _notion_headers():
    return {"Authorization": f"Bearer {NOTION_TOKEN}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json"}

def _manual_utf16_tab_read_bytes(b: bytes) -> pd.DataFrame:
    text = b.decode("utf-16")
    lines = [line.rstrip("\r\n") for line in text.splitlines()]
    rows = [line.split("\t") for line in lines]
    mx = max(len(r) for r in rows) if rows else 0
    rows = [r + [""] * (mx - len(r)) for r in rows]
    return pd.DataFrame(rows)

def read_input_table_from_bytes(name: str, data: bytes) -> pd.DataFrame:
    name_lower = name.lower()
    if name_lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data), header=None)

    head = data[:4]
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        try:
            return _manual_utf16_tab_read_bytes(data)
        except Exception:
            pass

    for enc in ["utf-8-sig", "cp949", "euc-kr", "utf-8"]:
        try:
            return pd.read_csv(io.BytesIO(data), header=None, sep=None, engine="python", encoding=enc)
        except Exception:
            continue

    for sep in ["\t", ",", ";"]:
        try:
            return pd.read_csv(io.BytesIO(data), header=None, sep=sep, encoding="utf-16")
        except Exception:
            continue

    raise RuntimeError(f"Failed to parse the input file: {name}")

def resolve_columns(df):
    headers = df.iloc[HEADER_ROW_IDX].astype(str).str.strip().tolist()
    body = df.iloc[HEADER_ROW_IDX + 1:].copy()
    body.columns = [str(c).strip() for c in headers]

    def find_col(cols, keyword):
        for c in cols:
            if c == keyword: return c
        for c in cols:
            if keyword in str(c): return c
        aliases = {
            "광고이름": ["광고 이름","광고명","Ad name","이미지 광고 이름"],
            "승인상태": ["승인 상태","Approval status"],
            "광고정책": ["광고 정책","광고정책 위반 사유","Policy"],
        }
        if keyword in aliases:
            for alias in aliases[keyword]:
                for c in cols:
                    if c == alias: return c
        raise KeyError(f"Missing required column: {keyword}")

    adname_col   = find_col(body.columns, "광고이름")
    approval_col = find_col(body.columns, "승인상태")
    policy_col   = find_col(body.columns, "광고정책")

    for c in [adname_col, approval_col, policy_col]:
        body[c] = body[c].astype(str).str.strip()
    return body, adname_col, approval_col, policy_col

def filter_step1(df, adname_col):
    def has_dot_before_first_underscore(s):
        if not isinstance(s,str) or s=="" or "_" not in s: return False
        return "." in s.split("_",1)[0]
    return df.loc[~df[adname_col].apply(has_dot_before_first_underscore)].copy()

def filter_step2(df, approval_col, policy_col):
    mask_A = (df[approval_col] == "승인됨")
    mask_B = (df[approval_col] == "승인됨(제한적)") & (df[policy_col] == TARGET_POLICY)
    return df.loc[mask_A | mask_B].copy()

def parse_fields_v5(name):
    if not isinstance(name,str) or name=="": return pd.Series([np.nan,np.nan,np.nan])
    parts = name.split("_")
    material = parts[3] if len(parts) >= 5 else np.nan
    size     = parts[-1] if len(parts) >= 2 else np.nan
    revision = parts[-2] if len(parts) >= 3 else np.nan
    if isinstance(revision,str) and isinstance(material,str) and revision == material:
        revision = "원본"
    return pd.Series([material,size,revision])

def build_step3(df12, adname_col, approval_col, policy_col):
    parsed = df12[adname_col].apply(parse_fields_v5)
    parsed.columns = ["소재명","사이즈","수정차수"]
    return pd.concat([parsed,
                      df12[[approval_col,policy_col]].rename(columns={approval_col:"승인상태",policy_col:"광고정책"})],
                     axis=1)

def pick_best_revision(series):
    series = series.dropna().astype(str)
    if (series == "원본").any(): return "원본"
    best=None
    for s in series:
        m = re.search(r"(\d+)\s*차", s) or re.search(r"(\d+)", s)
        if m:
            n=int(m.group(1))
            if best is None or n<best: best=n
    return f"{best}차" if best is not None else "x"

def extract_subjects_from_step1(df_step1, adname_col):
    parsed = df_step1[adname_col].apply(parse_fields_v5)
    material_col = parsed.iloc[:,0]
    seen, subjects = {}, []
    for v in material_col:
        if pd.notna(v) and str(v).strip()!="" and v not in seen:
            seen[v]=True; subjects.append(v)
    return subjects

def build_final_table(df3, subjects_override=None):
    df = df3.copy()
    for col in ["소재명","사이즈","수정차수"]:
        df[col] = df[col].astype(str).str.strip()
    df = df[df["사이즈"].isin(TARGET_SIZES)].copy()

    if subjects_override:
        subjects = [s for s in subjects_override if pd.notna(s) and str(s).strip()]
    else:
        seen, subjects = {}, []
        for name in df["소재명"]:
            if name not in seen:
                seen[name]=True; subjects.append(name)

    base = pd.DataFrame("x", index=TARGET_SIZES, columns=subjects)
    for subj in subjects:
        for sz in TARGET_SIZES:
            subset = df[(df["소재명"]==subj) & (df["사이즈"]==sz)]
            base.loc[sz,subj] = pick_best_revision(subset["수정차수"]) if len(subset) else "x"
    out_T = base.T.copy()
    out_T.index.name="소재명"; out_T.columns.name="사이즈"
    return out_T

def sync_final_table_to_notion(final_table: pd.DataFrame,
                               db_id: str,
                               subject_prop_name: str = "소재명",
                               size_props=("1x1","4x5","9x16","1920x1080"),
                               sleep_sec: float = 0.2):
    if not (NOTION_TOKEN and db_id): return
    def notion_get_database(db_id):
        r=requests.get(f"https://api.notion.com/v1/databases/{db_id}", headers=_notion_headers(), timeout=30)
        r.raise_for_status(); return r.json()
    def notion_update_page_properties(pid, updates):
        r=requests.patch(f"https://api.notion.com/v1/pages/{pid}", headers=_notion_headers(),
                         json={"properties":updates}, timeout=60)
        r.raise_for_status()
    def _get_plain_text(prop):
        if not isinstance(prop,dict): return ""
        t=prop.get("type")
        if t=="title":     return "".join(b.get("plain_text","") for b in prop.get("title",[]))
        if t=="rich_text": return "".join(b.get("plain_text","") for b in prop.get("rich_text",[]))
        return ""
    def notion_query_all_pages(db_id, page_size=100):
        url=f"https://api.notion.com/v1/databases/{db_id}/query"; cursor=None
        while True:
            payload={"page_size":page_size}
            if cursor: payload["start_cursor"]=cursor
            r=requests.post(url, headers=_notion_headers(), json=payload, timeout=60)
            r.raise_for_status()
            data=r.json()
            for row in data.get("results",[]): yield row
            if not data.get("has_more"): break
            cursor=data.get("next_cursor")

    subjects=[str(s) for s in list(final_table.index)]
    size_cols=[c for c in size_props if c in final_table.columns]
    dbjson=notion_get_database(db_id); db_props=dbjson.get("properties",{})
    size_types={}
    for col in size_cols:
        p=db_props.get(col)
        if p: size_types[col]=p.get("type")

    for page in notion_query_all_pages(db_id):
        props=page.get("properties",{}); subj=props.get(subject_prop_name)
        if subj is None: continue
        raw=_get_plain_text(subj).strip()
        if not raw: continue
        key_norm = raw.replace(" ","").strip()
        exact=None
        for s in subjects:
            if key_norm==s.replace(" ","").strip(): exact=s; break
        match = exact if exact is not None else (
            max([s for s in subjects if (s in raw) or (s.replace(" ","") in key_norm)], key=len)
            if any([(s in raw) or (s.replace(" ","") in key_norm) for s in subjects]) else None
        )
        if not match: continue
        row=final_table.loc[match]; updates={}
        for col in size_cols:
            val=row.get(col,"x"); val="x" if pd.isna(val) else val
            t=size_types.get(col)
            if t=="select":      updates[col]={"select":{"name":str(val)}}
            elif t=="multi_select": updates[col]={"multi_select":[{"name":str(val)}]}
            else:               updates[col]={"rich_text":[{"type":"text","text":{"content":str(val)}}]}
        if updates:
            try: notion_update_page_properties(page["id"], updates)
            except Exception:
                print("[Notion Sync] 실패\n", traceback.format_exc())
        time.sleep(sleep_sec)


# ====== 파일 하나 처리 -> (브랜드, 최종 테이블) 반환 ======
def process_single_file(file_name: str, file_bytes: bytes, do_sync: bool):
    df_raw = read_input_table_from_bytes(file_name, file_bytes)
    df_body, adname_col, approval_col, policy_col = resolve_columns(df_raw)
    df_step1 = filter_step1(df_body, adname_col)
    subjects_step1 = extract_subjects_from_step1(df_step1, adname_col)
    df_step12 = filter_step2(df_step1, approval_col, policy_col)
    df_step3 = build_step3(df_step12, adname_col, approval_col, policy_col)
    final_table = build_final_table(df_step3, subjects_override=subjects_step1)

    in_base = os.path.splitext(os.path.basename(file_name))[0]
    brand = in_base.split("_")[-1].strip() if "_" in in_base else in_base.strip()

    if do_sync and NOTION_TOKEN and NOTION_DB_ID:
        try:
            sync_final_table_to_notion(final_table, NOTION_DB_ID, "소재명",
                                       ("1x1","4x5","9x16","1920x1080"), 0.2)
        except Exception:
            print("[Notion Sync] 실패\n", traceback.format_exc())

    return brand, final_table


# ====== FastAPI ======
app = FastAPI(title="[구글] 콘텐츠 가용사이즈")

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print("[ERROR] Unhandled exception\n", tb)
    return PlainTextResponse(f"ERROR: {type(exc).__name__}: {exc}\n\n{tb}", status_code=500)

INDEX_HTML = """
<!doctype html><html lang="ko"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>[구글] 콘텐츠 가용사이즈</title>
<link href="https://unpkg.com/sakura.css/css/sakura.css" rel="stylesheet">
<style>.container{max-width:960px;margin:40px auto}.btn{background:#1e88e5;color:#fff;border:none;padding:10px 16px;border-radius:6px;cursor:pointer;width:100%} .card{background:#fff;border:1px solid #eee;border-radius:10px;padding:20px;box-shadow:0 2px 10px rgba(0,0,0,.05)}</style>
</head><body>
<div class="container">
  <h1>[구글] 콘텐츠 가용사이즈</h1>
  <div class="card">
    <input id="files" type="file" multiple />
    <div style="margin:10px 0"><label><input type="checkbox" id="notion"> Notion 동기화</label></div>
    <button id="run" class="btn">처리하기</button>
    <div id="status" style="margin-top:10px;color:#1e88e5;white-space:pre-line"></div>
  </div>
</div>
<script>
const filesEl=document.getElementById('files'); const run=document.getElementById('run'); const statusEl=document.getElementById('status'); const notion=document.getElementById('notion');
run.onclick=async ()=>{
  if(!filesEl.files.length){alert('파일을 선택하세요');return;}
  statusEl.textContent='처리 중...';
  const fd=new FormData(); for(const f of filesEl.files) fd.append('files', f);
  const res=await fetch('/process'+(notion.checked?'?notion_sync=true':''),{method:'POST',body:fd});
  if(!res.ok){ statusEl.textContent='에러:\\n'+await res.text(); return; }
  const blob=await res.blob();
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download='result.xlsx'; document.body.appendChild(a); a.click(); a.remove();
  statusEl.textContent='완료! 엑셀 다운로드됨';
};
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
def index(): return INDEX_HTML

@app.get("/healthz", response_class=PlainTextResponse)
def healthz(): return "ok"

# ==== 핵심: 항상 '엑셀 1개'만 내려줍니다 ====
@app.post("/process")
async def process(files: List[UploadFile] = File(...), notion_sync: bool = False):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    brands, tables = [], []
    for f in files:
        b = await f.read()
        brand, table = process_single_file(f.filename, b, do_sync=notion_sync)
        brands.append(brand)
        tables.append(table)

    uniq = sorted(set(brands))
    KST = timezone(timedelta(hours=9)); now_kst = datetime.now(KST)
    date_part = now_kst.strftime("%y%m%d"); time_part = now_kst.strftime("%H%M")

    if len(files) == 1:
        merged_brand = uniq[0]
    else:
        merged_brand = uniq[0] if len(uniq) == 1 else "복합"

    merged_stem = f"[{merged_brand}]_가용사이즈확인_{date_part}_{time_part}"

    merged_df = pd.concat(tables, axis=0) if len(tables) > 1 else tables[0]

    out = io.BytesIO()
    merged_df.to_excel(out, sheet_name="table", index=True)
    out.seek(0)

    filename = f"{merged_stem}.xlsx"
    filename_q = quote(filename)

    headers = {
        # filename* 하나만 사용 (UTF-8 안전)
        "Content-Disposition": f"attachment; filename*=UTF-8''{filename_q}",
        "Cache-Control": "no-store",
    }
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
