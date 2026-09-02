"""채점용 ask() 의 토큰 예산이 실제로 충분한지 확인한다.

gpt-5 계열은 `max_completion_tokens` 안에서 **추론 토큰을 먼저 쓴다.**
예산이 모자라면 예외 없이 `content=None` 이 돌아오고, judge_faithfulness() 는
그걸 `None`(판정 불가)으로 조용히 삼킨다. 그래서 눈으로는 안 보인다.

    python scripts/check_judge.py           # nano 로 10 vs 2000 비교
    python scripts/check_judge.py mini

API 호출 4번 든다.
"""

import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src"),
                str(Path(__file__).resolve().parents[1])]

from evaluation.generation import JUDGE_SYSTEM  # noqa: E402
from generation import DEFAULT_JUDGE_MAX_TOKENS, ask  # noqa: E402

USER = "\n".join([
    "[문서]", "사업기간은 계약일로부터 6개월이다.", "",
    "[질문]", "사업기간은?", "",
    "[답변]", "계약일로부터 6개월입니다.",
])


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "nano"
    print(f"모델 {model} · 현재 DEFAULT_JUDGE_MAX_TOKENS = {DEFAULT_JUDGE_MAX_TOKENS}\n")

    for budget in (DEFAULT_JUDGE_MAX_TOKENS, 2000):
        got = ask(model, JUDGE_SYSTEM, USER, max_tokens=budget).strip().upper()
        verdict = True if "YES" in got else False if "NO" in got else None
        print(f"  max_tokens={budget:>5}  응답={got!r:<12} 판정={verdict}")

    print("\n판정이 None 이면 채점이 안 되고 있는 것이다. 예외는 안 난다.")


if __name__ == "__main__":
    main()
