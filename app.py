# -*- coding: utf-8 -*-
import os, io, re, time, traceback
from datetime import datetime, timezone, timedelta
from typing import List
from urllib.parse import quote

import pandas as pd
import numpy as np
import requests

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, PlainTextResponse

HEADER_ROW_IDX = 2
TARGET_POLICY = "개인 맞춤 광고 정책 내 건강 관련 콘텐츠 (제한됨)"
TARGET_SIZES  = ["1x1", "4x5", "9x16", "1920x1080"]

NOTION_TOKEN   = os.getenv("NOTION_TOKEN", "")
NOTION_DB_ID   = os.getenv("NOTION_DB_ID", "")
NOTION_VERSION = "2022-06-28"

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
            if c == keyword:
                return c
        for c in cols:
            if keyword in str(c):
                return c
        aliases = {
            "광고이름": ["광고 이름","광고명","Ad name","이미지 광고 이름"],
            "승인상태": ["승인 상태","Approval status"],
            "광고정책": ["광고 정책","광고정책 위반 사유","Policy"],
        }
        if keyword in aliases:
            for alias in aliases[keyword]:
                for c in cols:
                    if c == alias:
                        return c
        raise KeyError(f"Missing required column: {keyword}")

    adname_col   = find_col(body.columns, "광고이름")
    approval_col = find_col(body.columns, "승인상태")
    policy_col   = find_col(body.columns, "광고정책")

    for c in [adname_col, approval_col, policy_col]:
        body[c] = body[c].astype(str).str.strip()
    return body, adname_col, approval_col, policy_col

def filter_step1(df, adname_col):
    def has_dot_before_first_underscore(s):
        if not isinstance(s,str) or s=="" or "_" not in s:
            return False
        return "." in s.split("_",1)[0]
    return df.loc[~df[adname_col].apply(has_dot_before_first_underscore)].copy()

def filter_step2(df, approval_col, policy_col):
    mask_A = (df[approval_col] == "승인됨")
    mask_B = (df[approval_col] == "승인됨(제한적)") & (df[policy_col] == TARGET_POLICY)
    return df.loc[mask_A | mask_B].copy()

def parse_fields_v5(name):
    if not isinstance(name,str) or name=="":
        return pd.Series([np.nan,np.nan,np.nan])
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
    if (series == "원본").any():
        return "원본"
    best=None
    for s in series:
        m = re.search(r"(\d+)\s*차", s) or re.search(r"(\d+)", s)
        if m:
            n=int(m.group(1))
            if best is None or n<best:
                best=n
    return f"{best}차" if best is not None else "x"

def extract_subjects_from_step1(df_step1, adname_col):
    parsed = df_step1[adname_col].apply(parse_fields_v5)
    material_col = parsed.iloc[:,0]
    seen, subjects = {}, []
    for v in material_col:
        if pd.notna(v) and str(v).strip()!="" and v not in seen:
            seen[v]=True
            subjects.append(v)
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
                seen[name]=True
                subjects.append(name)
    base = pd.DataFrame("x", index=TARGET_SIZES, columns=subjects)
    for subj in subjects:
        for sz in TARGET_SIZES:
            subset = df[(df["소재명"]==subj) & (df["사이즈"]==sz)]
            base.loc[sz,subj] = pick_best_revision(subset["수정차수"]) if len(subset) else "x"
    out_T = base.T.copy()
    out_T.index.name="소재명"; out_T.columns.name="사이즈"
    return out_T

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
    return brand, final_table

# ------------------ 앱 & 전역 에러 핸들러 ------------------
app = FastAPI(title="[구글] 콘텐츠 가용사이즈")

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print("[ERROR] Unhandled exception\n", tb)
    return PlainTextResponse(
        f"ERROR: {type(exc).__name__}: {exc}\n\n{tb}",
        status_code=500
    )
# ------------------------------------------------------------

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
    <div id="status" style="margin-top:10px;color:#1e88e5"></div>
  </div>
</div>
<script>
const filesEl=document.getElementById('files'); const run=document.getElementById('run'); const statusEl=document.getElementById('status'); const notion=document.getElementById('notion');
run.onclick=async ()=>{
  if(!filesEl.files.length){alert('파일을 선택하세요');return;}
  statusEl.textContent='처리 중...';
  const fd=new FormData(); for(const f of filesEl.files) fd.append('files', f);
  const res=await fetch('/process'+(notion.checked?'?notion_sync=true':''),{method:'POST',body:fd, cache:'no-store'});
  if(!res.ok){ statusEl.textContent='에러:\\n'+await res.text(); return; }
  const disp = res.headers.get('content-disposition') || '';
  // 기본 파일명
  let fname = 'result.xlsx';
  const m = /filename\\*=UTF-8''([^;]+)/i.exec(disp) || /filename="?([^";]+)"?/i.exec(disp);
  if(m && m[1]) fname = decodeURIComponent(m[1]);
  const blob=await res.blob();
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob); a.download=fname;
  document.body.appendChild(a); a.click(); a.remove();
  statusEl.textContent='완료! 엑셀 다운로드됨';
};
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
def index(): return INDEX_HTML

@app.get("/healthz", response_class=PlainTextResponse)
def healthz(): return "ok"

@app.post("/process")
async def process(files: List[UploadFile] = File(...), notion_sync: bool = False):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    brands, dfs = [], []
    for f in files:
        b = await f.read()
        brand, final_table = process_single_file(f.filename, b, do_sync=notion_sync)
        brands.append(brand)
        dfs.append(final_table)

    uniq = sorted(set(brands))
    KST = timezone(timedelta(hours=9)); now_kst = datetime.now(KST)
    date_part = now_kst.strftime("%y%m%d"); time_part = now_kst.strftime("%H%M")

    # 1개면 해당 브랜드, 여러 개면 단일/복수 브랜드에 따라 이름 지정
    if len(files) == 1:
        merged_brand = uniq[0]
    else:
        merged_brand = uniq[0] if len(uniq) == 1 else "복합"

    merged_stem = f"[{merged_brand}]_가용사이즈확인_{date_part}_{time_part}"

    # 통합 테이블 생성
    merged_df = pd.concat(dfs, axis=0) if len(dfs) > 1 else dfs[0]
    out = io.BytesIO()
    merged_df.to_excel(out, sheet_name="table", index=True)
    out.seek(0)

    # 파일명(UTF-8 안전)
    utf8_name = f"{merged_stem}.xlsx"
    utf8_name_q = quote(utf8_name)

    headers = {
        # 확실히 xlsx로 저장되게 이름을 지정(캐시 방지 헤더도 포함)
        "Content-Disposition": f"attachment; filename=result.xlsx; filename*=UTF-8''{utf8_name_q}",
        "Cache-Control": "no-store"
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
