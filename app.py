#대소문자 구분반영
#보라샷 위한 . 2개이상으로 필터조건 변경
#노션 연동시 이름 정확히일치로 코드 수정함(09/05)

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

# ====== '.'이 2개 이상일 때만 제외 ======
def filter_step1(df, adname_col):
    """
    첫 번째 언더스코어(_) 이전 구간에서 점(.)이 '2개 이상'일 때만 제외.
    - 문자열이 아니거나 빈 문자열, 혹은 '_'가 없는 경우: 제외하지 않음(False)
    """
    def has_two_or_more_dots_before_first_underscore(s):
        if not isinstance(s, str) or s == "" or "_" not in s:
            return False
        prefix = s.split("_", 1)[0]
        return prefix.count(".") >= 2

    return df.loc[~df[adname_col].apply(has_two_or_more_dots_before_first_underscore)].copy()

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
                      df12[[approval_col,policy_col]].re
