"""전처리팀 청크 jsonl 이 쓸 수 있는 상태인지 본다. 인덱스 만들기 전에 돌린다.

    python scripts/retrieval/check_chunks.py data/processed/cleaned_documents_v6__chunks_1500_250.jsonl

실패하면 그 줄 번호를 찍고 멈춘다. 통과하면 길이 분포를 보여준다 —
표 원자성 때문에 1500 을 넘는 청크가 얼마나 되는지가 여기서 보인다.
"""

import json
import sys
from collections import Counter

SIZE = 1500

path = sys.argv[1]
lens, over, docs = [], [], Counter()

for n, line in enumerate(open(path, encoding="utf-8"), 1):
    d = json.loads(line)
    search, gen = d["page_content"], d["page_content_for_generation"]
    meta = d["metadata"]

    assert search, f"{n}줄: page_content 가 비었다"
    assert gen, f"{n}줄: page_content_for_generation 이 비었다"
    # 마커를 지운 쪽이 더 길면 자르기·지우기 순서가 뒤바뀐 것이다
    assert len(search) <= len(gen), f"{n}줄: 검색용({len(search)})이 생성용({len(gen)})보다 길다"
    # 보이지 않는 경계 마커가 새면 여기서 잡힌다
    assert not any(c in search or c in gen for c in "￼"), f"{n}줄: 경계 마커가 샜다"
    assert "chunk_index" in meta, f"{n}줄: chunk_index 없음"

    lens.append(len(search))
    if len(search) > SIZE:
        over.append((n, len(search)))
    docs[meta.get("source")] += 1

lens.sort()
print(f"청크 {len(lens):,}개 · 문서 {len(docs)}건")
print(f"검색용 길이  중앙값 {lens[len(lens)//2]}  최대 {lens[-1]}")
print(f"{SIZE} 초과: {len(over)}개 ({len(over)/len(lens)*100:.1f}%)")
for n, size in sorted(over, key=lambda x: -x[1])[:5]:
    print(f"  {n}줄 {size:,}자")
if over:
    print(f"\n6,000자 예산에 {6000//lens[-1]}개까지밖에 안 들어가는 청크가 있다.")
