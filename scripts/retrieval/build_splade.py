"""Splade 희소 인덱스를 미리 만들어 둔다. **GPU 가 있는 VM 에서 돌린다.**

TEI 로는 안 된다. 한국어 Splade 모델(PIXIE, splade-ko)이 전부 ModernBERT 인데
TEI 의 splade pooling 은 ModernBert 를 지원하지 않는다:

    `splade` is not supported for ModernBert

그래서 모델을 직접 올려야 하는데, 코퍼스 9,200개를 인코딩하는 게 비싸다.
맥에서 돌리면 GPU 가 97% 에 90도가 넘는다. **VM 에서 한 번 만들어 npz 를
가져오면 끝난다.** 검색할 때 필요한 건 질문 한 줄 인코딩뿐이라 그건 맥에서
돌아도 열이 안 난다.

    # VM 에서
    python scripts/retrieval/build_splade.py --chunks cleaned_documents_v3__recursive_1200_200

    # 만들어진 것을 맥으로
    scp vm:~/rfp-rag-system/outputs/vectorstore/*__splade__*.npz outputs/vectorstore/
"""

import argparse
import os
import sys
from pathlib import Path

# 조각난 메모리 때문에 큰 덩어리를 못 잡는 일을 줄인다. torch 를 부르기 전에 둬야 한다.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import chunking  # noqa: E402
from pieces import Splade, SpladeModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", required=True, help="outputs/chunks 의 청크 이름")
    parser.add_argument("--model", default=SpladeModel.PIXIE.value)
    parser.add_argument(
        "--batch", type=int, default=8,
        help="한 묶음 크기. 메모리가 모자라면 알아서 절반씩 줄인다"
    )
    parser.add_argument(
        "--max-length", type=int, default=1024, help="자를 토큰 수"
    )
    args = parser.parse_args()

    chunks = chunking.load_chunks(args.chunks)
    Splade(
        chunks,
        model=args.model,
        batch_size=args.batch,
        max_length=args.max_length,
        cache=args.chunks,
        verbose=True,
    )
    print("끝. outputs/vectorstore/ 의 npz 를 맥으로 옮기면 된다.")


if __name__ == "__main__":
    main()
