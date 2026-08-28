
from ..extract import DATA_DIR
from ..pipeline import print_a2_result, run_a2

df_a2, texts, square_lines, heading_candidates = run_a2(
    DATA_DIR,
)

print_a2_result(
    df_a2,
    texts,
    square_lines,
    heading_candidates,
)
