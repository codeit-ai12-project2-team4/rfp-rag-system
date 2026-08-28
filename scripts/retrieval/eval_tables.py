#!/usr/bin/env python
"""표 추출이 얼마나 잘 됐는지 잰다.

    python scripts/eval_tables.py                     정답 없이 (전체 문서)
    python scripts/eval_tables.py --limit 10          10건만
    python scripts/eval_tables.py --gold data/tables_gold.jsonl   정답 대조

## 정답 없이 재는 것 — 세 가지

1. 격자복원률   HWP 가 "17행 4열" 이라고 적어 둔 표를 그대로 복원했나.
                실패하면 값만 늘어놓는다(fallback). 낮으면 병합 파싱이 틀린 것.
2. 격자채움률   복원한 격자에서 글자가 든 칸의 비율. 병합이 많으면 원래
                100% 가 안 나온다. **문서끼리 비교할 때 의미가 있다.**
3. 줄보존률     문단만 뽑는 hwp.py 의 결과가 표 복원본에 다 남아 있나.
                표를 펴다가 글자를 흘리면 여기서 잡힌다. 1.0 에 가까워야 한다.

셋 다 정답이 필요 없다. 대신 **"틀린 값을 만들었나"는 못 잡는다.**
그건 아래 정답 대조로 본다.

## 정답 대조 — 행 단위로 20~30줄만

표 추출의 존재 이유는 글자가 아니라 **행과 열 관계**다. 글자는 hwp.py 도
다 가져온다. 그러니 정답도 행 단위로 적는다. 한글에서 파일 몇 개 열고
중요한 표(평가배점표, 요구사항표)의 행을 그대로 옮겨 적으면 된다.

    data/tables_gold.jsonl
    {"file_name": "...hwp", "row": ["기술능력평가", "80"]}
    {"file_name": "...hwp", "row": ["가격평가", "20"]}

한 줄 안에 그 행의 값이 **다 같이** 나오면 맞은 것이다. 흩어져 있으면
행 관계가 깨진 것이고, 그게 이 파서가 막으려던 실패다.
"""

import argparse
import json
import re
import unicodedata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # 평평한 import: chunking, preprocessing …
sys.path.insert(0, str(ROOT))           # config.settings

import pandas as pd

from config import settings
from preprocessing import load_metadata
from preprocessing.hwp import extract_hwp_text
from preprocessing.hwp_table import extract_with_report


def lost_lines(plain, tabled, min_len=1):
    """문단 추출기엔 있는데 표 복원본에선 사라진 줄. (잃은줄, 전체줄수)

    집합으로 두 번 거르고 남은 것만 부분문자열로 찾는다. 전부 부분문자열로
    찾으면 문서당 줄수×글자수라 100건에 몇 분 걸린다.
    """
    lines = {ln.strip() for ln in plain.splitlines() if len(ln.strip()) >= min_len}
    if not lines:
        return [], 0
    exact = {ln.strip() for ln in tabled.splitlines()}       # 표 밖 줄은 그대로 남는다
    tokens = {t.strip() for t in re.split(r"[=·|:\n]", tabled)}  # 표를 펴며 쓴 구분자로 되쪼갠다
    rest = [ln for ln in lines if ln not in exact and ln not in tokens]
    return sorted(ln for ln in rest if ln not in tabled), len(lines)


def line_keep_rate(plain, tabled, min_len=10):
    """문단 추출기의 줄이 표 복원본에 몇 %나 남아 있나."""
    lost, total = lost_lines(plain, tabled, min_len)
    return round(1 - len(lost) / total, 3) if total else None


def measure(path):
    """파일 하나. 실패하면 error 만 담아 돌려준다."""
    row = {"file_name": path.name}
    collected = []
    try:
        tabled, report = extract_with_report(path, collect=collected)
        plain = extract_hwp_text(path)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row, ""

    # 렌더까지 해 놓고 최종 본문에 안 들어간 표. 표 안의 표가 이렇게 사라진다.
    lost = [t for t in collected if t["rendered"] and t["rendered"] not in tabled]

    tables = report["tables"]
    row.update({
        "표수": tables,
        "칸수": report["cells"],
        "격자복원률": round(1 - report["fallback"] / tables, 3) if tables else None,
        "격자채움률": round(report["filled"] / report["slots"], 3) if report["slots"] else None,
        "빈표": report["empty"],
        "유실표": len(lost),
        "유실글자": sum(len(t["rendered"]) for t in lost),
        "빈칸률": round(report["blank_cells"] / report["cells"], 3) if report["cells"] else None,
        "줄보존률": line_keep_rate(plain, tabled),          # 10자 이상 줄만
        "짧은줄보존률": line_keep_rate(plain, tabled, min_len=1),  # 숫자·코드 포함
        "글자수": len(tabled),
        "문단만글자수": len(plain),
    })
    return row, tabled


def check_gold(gold_path, texts):
    """정답 행이 한 줄 안에 다 같이 나오는지."""
    rows = []
    for line in Path(gold_path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = json.loads(line)
        text = texts.get(item["file_name"])
        if text is None:
            rows.append({**item, "결과": "문서없음"})
            continue
        values = [str(v).strip() for v in item["row"] if str(v).strip()]
        together = any(all(v in ln for v in values) for ln in text.splitlines())
        anywhere = all(v in text for v in values)
        rows.append({
            "file_name": item["file_name"][:30],
            "row": " · ".join(values)[:50],
            "결과": "행복원" if together else ("글자만" if anywhere else "없음"),
        })
    return pd.DataFrame(rows)


def show_one(pattern, names):
    """문서 하나를 눈으로 본다. 지표로 못 잡는 건 결국 이걸로 봐야 한다."""
    hit = [n for n in names if pattern in n]
    if not hit:
        print(f"'{pattern}' 이 들어간 파일이 없습니다.")
        return
    name = hit[0]
    tabled, report = extract_with_report(settings.RAW / name)
    plain = extract_hwp_text(settings.RAW / name)
    lost, total = lost_lines(plain, tabled)

    print(f"{name}")
    print(f"  표 {report['tables']}개 · 빈표 {report['empty']}개 · 칸 {report['cells']}개 "
          f"· 빈칸 {report['blank_cells']}개")

    print(f"\n[잃은 줄] {len(lost)}/{total} — 문단엔 있는데 표 복원본엔 없다")
    for line in lost[:30]:
        print(f"   {line[:100]}")
    if len(lost) > 30:
        print(f"   … 외 {len(lost) - 30}줄")

    print("\n[표 복원 결과] 앞 15줄 — 머리글=값 짝이 말이 되는지 본다")
    rendered = [ln for ln in tabled.splitlines() if ln.startswith("[")]
    for line in rendered[:15]:
        print(f"   {line[:150]}")


MAX_CELL = 20  # 칸 하나를 이 폭까지만 보여준다 (한글 기준 10자)


def _width(text):
    """한글은 두 칸, 영문은 한 칸. 격자를 눈으로 맞추려면 이게 필요하다."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _cell(text):
    text = " ".join(text.split())  # 칸 안 줄바꿈은 공백으로
    while _width(text) > MAX_CELL:
        text = text[:-1]
    return text


def _pad(text, width):
    return text + " " * max(width - _width(text), 0)


def dump_tables(pattern, names, out_dir):
    """표를 격자 그대로 텍스트 파일에 뽑는다.

    **한글 창을 왼쪽에, 이 파일을 오른쪽에 놓고 훑으라고 만든 것이다.**
    화면에서 보는 게 격자니까 격자로 보여줘야 대조가 된다.
    렌더 결과(→)는 그 표가 실제로 검색 인덱스에 어떻게 들어가는지다.
    """
    hit = [n for n in names if pattern in n]
    if not hit:
        print(f"'{pattern}' 이 들어간 파일이 없습니다.")
        return
    name = hit[0]

    tables = []
    text, report = extract_with_report(settings.RAW / name, collect=tables)
    for t in tables:
        t["lost"] = bool(t["rendered"]) and t["rendered"] not in text

    lines = [
        name,
        f"표 {report['tables']}개 · 칸 {report['cells']}개 · "
        f"빈칸 {report['blank_cells']}개 · 빈표 {report['empty']}개 · "
        f"유실표 {sum(1 for t in tables if t['lost'])}개",
        "",
        "한글에서 같은 파일을 열고 표를 순서대로 대조하세요.",
        "  격자      = 파서가 복원한 행×열. 한글 화면과 모양이 같아야 합니다.",
        "  → 렌더    = 이 표가 검색 인덱스에 들어가는 실제 문장입니다.",
        "  격자복원실패 = 병합을 못 읽어 값만 늘어놓은 표입니다.",
        "  ⚠ 사라짐   = 렌더까지 했는데 최종 본문에 안 들어간 표입니다. 버그입니다.",
        "=" * 100,
        "",
    ]

    for i, t in enumerate(tables, 1):
        flag = "  ⚠ 최종 본문에서 사라짐" if t["lost"] else ""
        head = (f"── 표 {i}  ({t['rows']}행 {t['cols']}열, 칸 {t['cells']}개)  "
                f"{t['caption']}{flag}")
        lines.append(head + " " + "─" * max(100 - _width(head), 0))

        if t["grid"] is None:
            lines.append("   [격자복원실패] 병합을 못 읽었습니다")
        else:
            grid = [[_cell(c) for c in row] for row in t["grid"]]
            widths = [max((_width(r[c]) for r in grid), default=0) for c in range(t["cols"])]
            for row in grid:
                lines.append("   " + " | ".join(_pad(v, w) for v, w in zip(row, widths)).rstrip())

        rendered = (t["rendered"] or "").splitlines()
        lines.append("")
        for r in rendered[:6]:
            lines.append(f"   → {r}")
        if len(rendered) > 6:
            lines.append(f"   → … 외 {len(rendered) - 6}줄")
        if not rendered:
            lines.append("   → (빈 표 — 글자를 하나도 못 건졌습니다)")
        lines.append("")

    out = Path(out_dir) / f"표_{Path(name).stem[:30]}.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"표 {len(tables)}개 → {out}")
    print("한글에서 같은 파일을 열고 나란히 놓고 보세요.")


def report(df, baseline=None):
    """사람이 읽고 판단할 수 있게 찍는다. 숫자 나열이 아니라 판정문."""
    ok = df[df["error"].isna()] if "error" in df else df
    failed = df[df["error"].notna()] if "error" in df else df.iloc[:0]

    def avg(col):
        return ok[col].dropna().mean()

    print(f"\n표 추출 점검 — HWP {len(df)}건 · 성공 {len(ok)} · 실패 {len(failed)}")
    print(f"  표 {int(ok['표수'].sum()):,}개 · 칸 {int(ok['칸수'].sum()):,}개")
    if len(failed):
        print(failed[["file_name", "error"]].to_string(index=False))

    grid, blank, fill = avg("격자복원률"), avg("빈칸률"), avg("격자채움률")
    print("\n[구조] 격자를 복원했나")
    print(f"  격자복원률 {grid:.3f}   병합 파싱이 깨진 표 "
          f"{int((1 - ok['격자복원률'].fillna(1)).mul(ok['표수']).sum()):,}개")
    print( "             ※ '예외가 안 났다'는 뜻이지 '격자가 맞다'는 뜻이 아니다")
    print(f"  빈칸률     {blank:.3f}   문서가 선언한 칸의 {blank:.0%}가 원래 빈 칸")
    print(f"  격자채움률 {fill:.3f}   빈칸률과 합이 {fill + blank:.2f}", end="")
    print("  → 격자의 빈자리가 빈 칸으로 설명된다  OK"
          if 0.9 <= fill + blank <= 1.1 else "  → 설명이 안 된다. 칸을 흘리고 있다")

    keep = avg("짧은줄보존률")
    worst = ok.nsmallest(1, "짧은줄보존률").iloc[0]
    n_lost, c_lost = int(ok["유실표"].sum()), int(ok["유실글자"].sum())
    docs_lost = int((ok["유실표"] > 0).sum())

    print("\n[손실] 글자를 흘렸나")
    if n_lost:
        print(f"  유실표       {n_lost:,}개   렌더까지 해 놓고 최종 본문에 안 들어간 표"
              f" ({c_lost:,}자)")
        top = ok.nlargest(3, "유실글자")[["file_name", "유실표", "유실글자"]]
        print(f"               {docs_lost}개 문서에서 발생. 표 안의 표가 부모에서 버려진 것이다.")
        for _, r in top.iterrows():
            if r["유실표"]:
                print(f"                 {r['file_name'][:38]}  표 {int(r['유실표'])}개 "
                      f"{int(r['유실글자']):,}자")
    else:
        print("  유실표       0개   OK — 표가 통째로 사라지는 일은 없다")
    print(f"  짧은줄보존률 {keep:.3f}   문단 추출기가 뽑은 줄의 {keep:.1%}가 표 복원본에도 있다")
    print(f"               최악 {worst['짧은줄보존률']:.3f}  {worst['file_name'][:40]}")
    print(f"  빈표         {int(ok['빈표'].sum()):,}개   표인데 글자를 하나도 못 건짐 "
          f"({ok['빈표'].sum() / max(ok['표수'].sum(), 1):.1%})")

    print("\n[정확] 값이 맞나")
    print("  측정 안 됨 — 열이 한 칸 밀려도 위 지표는 전부 만점이다.")
    print("  data/tables_gold.jsonl 에 실제 표 행 20줄을 적고 --gold 로 재라.")

    print("\n손볼 문서 5건 (짧은줄보존률 낮은 순)")
    cols = ["file_name", "표수", "빈칸률", "짧은줄보존률"]
    low = ok.nsmallest(5, "짧은줄보존률")[cols]
    print(low.assign(file_name=low["file_name"].str[:38]).to_string(index=False))

    if baseline:
        compare_baseline(ok, baseline)


def compare_baseline(ok, baseline, col="짧은줄보존률", eps=0.002):
    """이전 CSV 와 문서별로 비교한다. **평균만 보면 상쇄돼서 안 보인다.**"""
    before = pd.read_csv(baseline)[["file_name", col]].rename(columns={col: "before"})
    m = ok[["file_name", col]].rename(columns={col: "after"}).merge(before, on="file_name")
    if m.empty:
        print(f"\n[변화] {baseline} 와 겹치는 문서가 없다")
        return
    m["차이"] = (m["after"] - m["before"]).round(4)

    up = int((m["차이"] > eps).sum())
    down = int((m["차이"] < -eps).sum())
    same = len(m) - up - down
    print(f"\n[변화] {Path(baseline).name} 대비 · {col}")
    print(f"  좋아짐 {up}건 · 나빠짐 {down}건 · 그대로 {same}건 "
          f"(±{eps} 이내는 그대로로 본다)")
    print(f"  평균 {m['차이'].mean():+.4f}", end="")
    print("  → 평균은 상쇄된다. 아래 문서별 증감을 봐라." if abs(m["차이"].mean()) < eps
          else "")

    m = m.reindex(m["차이"].abs().sort_values(ascending=False).index)
    top = m.head(8).assign(file_name=lambda d: d["file_name"].str[:36])
    print("\n  가장 크게 변한 8건")
    print(top.to_string(index=False))
    if down:
        print(f"\n  ⚠ 나빠진 {down}건은 패치가 망가뜨린 표다. "
              f"--show 로 하나 열어 확인할 것.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gold")
    parser.add_argument("--show", help="파일명 일부. 그 문서 하나만 눈으로 본다")
    parser.add_argument("--baseline", help="이전 table_report.csv. 문서별 증감을 본다")
    parser.add_argument("--dump", help="파일명 일부. 그 문서의 표를 격자로 파일에 뽑는다")
    parser.add_argument("--out", default=str(settings.REPORTS / "table_report.csv"))
    args = parser.parse_args()

    meta = load_metadata()
    names = [n for n in meta["file_name"] if str(n).lower().endswith(".hwp")]

    if args.dump:
        settings.make_dirs()
        dump_tables(args.dump, names, settings.REPORTS)
        return

    if args.show:
        show_one(args.show, names)
        return

    if args.limit:
        names = names[: args.limit]
    if args.gold:  # 정답에 적힌 문서는 --limit 과 무관하게 반드시 넣는다
        want = {json.loads(ln)["file_name"]
                for ln in Path(args.gold).read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")}
        names += [n for n in want if n not in names]

    rows, texts = [], {}
    for i, name in enumerate(names, 1):
        path = settings.RAW / str(name)
        if not path.exists():
            continue
        row, text = measure(path)
        rows.append(row)
        texts[name] = text
        print(f"  {i}/{len(names)} {name[:40]}", end="\r")

    df = pd.DataFrame(rows)
    print(" " * 70, end="\r")
    df.to_csv(args.out, index=False, encoding="utf-8-sig")

    report(df, args.baseline)
    print(f"\n표별 수치: {args.out}")

    if args.gold:
        print("\n[정답 대조] 행이 한 줄 안에 같이 나오나")
        gold = check_gold(args.gold, texts)
        print(gold.to_string(index=False))
        counts = gold["결과"].value_counts()
        n = len(gold)
        print(f"\n  행복원 {counts.get('행복원', 0)}/{n}  "
              f"글자만 {counts.get('글자만', 0)}  없음 {counts.get('없음', 0)}")


if __name__ == "__main__":
    main()
