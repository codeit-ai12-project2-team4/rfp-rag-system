"""전처리팀 청크 jsonl 이 쓸 수 있는 상태인지 본다. 인덱스 만들기 전에 돌린다.

    python scripts/retrieval/check_chunks.py outputs/chunks/chunks_cleaned_documents_v7__1500_250.jsonl

실패하면 그 줄 번호를 찍고 멈춘다. 통과하면 길이 분포를 보여준다 —
표 원자성 때문에 1500 을 넘는 청크가 얼마나 되는지가 여기서 보인다.

검색용이 생성용보다 길 수 있다: `_flatten_table_block()` 이 데이터 행마다
헤더를 되풀이해 붙이기 때문이다. 그래서 길이 대소는 검사하지 않는다.
"""

import json
import sys
from collections import Counter

SIZE = 1500
LEFTOVERS = ("|", "<br>", "[행 ", "[표 복원", "[표 부분")

path = sys.argv[1]
lens, over, docs, dirty = [], [], Counter(), []

for n, line in enumerate(open(path, encoding="utf-8"), 1):
    d = json.loads(line)
    search, gen = d["page_content"], d["page_content_for_generation"]
    meta = d["metadata"]

    assert search.strip(), f"{n}줄: page_content 가 비었다"
    assert gen.strip(), f"{n}줄: page_content_for_generation 이 비었다"
    assert "chunk_index" in meta, f"{n}줄: chunk_index 없음"
    assert meta.get("source"), f"{n}줄: source 없음"

    found = [mark for mark in LEFTOVERS if mark in search]
    if found:
        dirty.append((n, found))

    lens.append(len(search))
    if len(search) > SIZE:
        over.append((n, len(search)))
    docs[meta["source"]] += 1

lens.sort()
print(f"청크 {len(lens):,}개 · 문서 {len(docs)}건 · 문서당 평균 {len(lens)/len(docs):.0f}개")
print(f"검색용 길이  중앙값 {lens[len(lens)//2]}  평균 {sum(lens)//len(lens)}  최대 {lens[-1]:,}")
print(f"{SIZE} 초과 {len(over)}개 ({len(over)/len(lens)*100:.1f}%)")
for n, size in sorted(over, key=lambda x: -x[1])[:5]:
    print(f"    {n}줄 {size:,}자")

if dirty:
    print(f"\n⚠ 검색용에 표 마크업이 남은 청크 {len(dirty)}개")
    for n, marks in dirty[:5]:
        print(f"    {n}줄 {marks}")
else:
    print("\n검색용에 표 마크업 잔존 없음")

budget = 6000
print(f"\n{budget:,}자 예산에 들어갈 청크 수: 중앙값 기준 {budget//lens[len(lens)//2]}개"
      f" · 최대 청크만 담으면 {max(1, budget//lens[-1])}개")

# --- 로더까지 확인한다. langchain 이 있는 데서만 돈다 -----------------------
try:
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
    import chunking
except ImportError as error:
    print(f"\n(로더 확인 건너뜀: {error})")
else:
    name = Path(path).name.removeprefix("chunks_").removesuffix(".jsonl")
    loaded = chunking.load_chunks(name)
    assert len(loaded) == len(lens), f"로더가 {len(loaded)}개를 읽었다 (파일은 {len(lens)}개)"
    for key in ("doc_id", "title", "agency", "chunk_id", "gen"):
        have = sum(bool(d.metadata.get(key)) for d in loaded)
        flag = "O" if have == len(loaded) else "X"
        print(f"  {flag} metadata['{key}']  {have:,}/{len(loaded):,}")
    print(f"\n예시  doc_id={loaded[0].metadata['doc_id']}  "
          f"title={str(loaded[0].metadata.get('title'))[:30]}")
