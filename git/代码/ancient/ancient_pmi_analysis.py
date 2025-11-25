# ancient_pmi_analysis.py
"""
古代语料 PMI 分析脚本（句子级共现）
说明：
- 输入文件：ancient_data.xlsx，包含列名 "分词文本"（空格分词，每行一句）
- 停用词：stopwords.txt（每行一个词/标点）
- 输出：PMI_ancient.xlsx、PMI_top_<N>.csv、可选图像文件
"""

import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------- 参数区（可修改） -----------------------------
INPUT_XLSX = "ancient_corpus.xlsx"   # 输入文件
TOKEN_COL = "分词文本"             # 原始分词列
FILTERED_COL = "tokens_no_stop"    # 新列名（不改原列）
STOPWORDS_FILE = "stopwords_cn.txt"   # 停用词表（每行一个）
OUTPUT_PREFIX = "ancient_out"      # 输出文件前缀
TARGET = "革命"                    # 中心词
MIN_COOC_FREQ = 2                  # 共现最小频次阈值（古代建议 >=2）
REMOVE_PUNCT = True                # 是否把标点当作停用词（推荐 True）
TOP_N_PLOT = 25                    # 输出可视化时取前 N
CHAR_LEVEL = True                 # 是否运行字符级补充统计（True/False）
CHAR_MODE = "strict"               # "strict" (仅 token == char) 或 "contains" (token 包含该字符)
SMOOTH = 1.0                       # 平滑常数，用于分母/分子平滑 (可设 1.0)
# ---------------------------------------------------------------------------

def load_stopwords(path, remove_punct=True):
    s = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                w = line.strip()
                if w:
                    s.add(w)
    if remove_punct:
        puncts = {"，", "。", "、", "；", "：", "？", "！", ",", ".", ":", ";", "(", ")", "（", "）",
                  "“", "”", "\"", "'", "‘", "’", "—", "–", "…", "·", "【", "】"}
        s |= puncts
    return s

def filter_tokens_str(token_str, stopwords):
    toks = [t for t in token_str.split() if t and t not in stopwords]
    return " ".join(toks)

def compute_counts(token_series, target, stopwords):
    """
    token_series: iterable of token strings (space-separated) where stopwords already removed
    returns:
      N: total sentences
      count_target: number of sentences containing target
      count_word: Counter of sentences containing each word (token-level)
      count_co: Counter of cooccurrence counts with target (per sentence)
    """
    N = 0
    count_target = 0
    count_word = Counter()
    count_co = Counter()

    for s in token_series:
        N += 1
        tokens = s.split() if isinstance(s, str) else []
        token_set = set(tokens)  # 按句计算，避免句内重复多计
        # 统计词出现句数
        for w in token_set:
            count_word[w] += 1
        # 若包含 target，统计共现
        if target in token_set:
            count_target += 1
            for w in token_set:
                if w != target:
                    count_co[w] += 1
    return N, count_target, count_word, count_co

def compute_pmi_table(N, count_target, count_word, count_co, min_freq=2, smooth=1.0):
    """
    PMI(w, target) = log2( (co * N + smooth) / ((count_target + smooth) * (count_word[w] + smooth)) )
    返回 DataFrame（word, co, word_count, pmi）
    """
    rows = []
    for w, co in count_co.items():
        if co < min_freq:
            continue
        wcount = count_word.get(w, 0)
        numerator = (co * N) + smooth
        denominator = (count_target + smooth) * (wcount + smooth)
        pmi = math.log2(numerator / denominator) if denominator > 0 else float("-inf")
        rows.append((w, co, wcount, pmi))
    df = pd.DataFrame(rows, columns=["word", "cooc_count", "word_count", "pmi"])
    df = df.sort_values("pmi", ascending=False).reset_index(drop=True)
    return df

def char_level_stats(token_series, target, chars_of_interest, mode="strict"):
    """
    对若干关注字符做字符级统计（strict 或 contains）
    mode:
      - "strict": 仅 token == char 时计入（即句内有精确字符 token）
      - "contains": token 中包含该字符则视为出现（用于捕获复合词含字符情况）
    返回 DataFrame：char, sentences_with_char, cooc_with_target, cooc_rate
    """
    char_counts = {c: 0 for c in chars_of_interest}
    char_cooc = {c: 0 for c in chars_of_interest}
    N = 0
    for s in token_series:
        N += 1
        tokens = s.split() if isinstance(s, str) else []
        token_set = set(tokens)
        for c in chars_of_interest:
            found = False
            if mode == "strict":
                if c in token_set:
                    found = True
            elif mode == "contains":
                # any token contains c
                for tok in token_set:
                    if c in tok:
                        found = True
                        break
            if found:
                char_counts[c] += 1
                if target in token_set:
                    char_cooc[c] += 1
    rows = []
    for c in chars_of_interest:
        cnt = char_counts[c]
        co = char_cooc[c]
        rate = co / cnt if cnt > 0 else 0.0
        rows.append((c, cnt, co, rate))
    df = pd.DataFrame(rows, columns=["char", "sent_count", "cooc_with_target", "cooc_rate"])
    df = df.sort_values("cooc_with_target", ascending=False).reset_index(drop=True)
    return df

def plot_top_pmi(df_pmi, topn=20, out_prefix="pmi_plot"):
    df = df_pmi.head(topn)
    if df.empty:
        return
    plt.figure(figsize=(8, max(4, 0.25 * len(df))))
    plt.barh(df['word'][::-1], df['pmi'][::-1])
    plt.xlabel("PMI (log2)")
    plt.title(f"Top {topn} PMI words with '{TARGET}'")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}.png", dpi=200)
    plt.close()

def main():
    # 加载数据
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"未找到输入文件：{INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX, engine="openpyxl")
    if TOKEN_COL not in df.columns:
        raise ValueError(f"输入文件缺少列：{TOKEN_COL}")

    # 加载停用词
    stopwords = load_stopwords(STOPWORDS_FILE, remove_punct=REMOVE_PUNCT)

    # 生成过滤后的新列（不改原列）
    df[FILTERED_COL] = df[TOKEN_COL].astype(str).apply(lambda s: filter_tokens_str(s, stopwords))

    # 按需：可以按"时期/大时期"分组，这里假设全体统一（你后续可以按时期 groupby）
    token_series = df[FILTERED_COL].tolist()

    # 统计基础
    N, count_target, count_word, count_co = compute_counts(token_series, TARGET, stopwords)

    print(f"总句子数 N = {N}")
    print(f"包含 '{TARGET}' 的句子数 = {count_target}")
    print(f"候选共现词（去停用词后）数量 = {len(count_co)}")

    # 计算 PMI 表
    pmi_df = compute_pmi_table(N, count_target, count_word, count_co,
                               min_freq=MIN_COOC_FREQ, smooth=SMOOTH)

    # 保存结果
    # 1. 完整的 PMI 结果（Excel格式）
    out_pmi_xlsx = f"{OUTPUT_PREFIX}_PMI_full.xlsx"
    pmi_df.to_excel(out_pmi_xlsx, index=False)
    print(f"完整 PMI 结果已保存：{out_pmi_xlsx}")

    # 2. 完整的 PMI 结果（CSV格式，用于其他分析）
    out_pmi_csv = f"{OUTPUT_PREFIX}_PMI_full.csv"
    pmi_df.to_csv(out_pmi_csv, index=False, encoding="utf-8-sig")
    print(f"完整 PMI 结果(CSV)已保存：{out_pmi_csv}")

    # 3. 前 N 个结果的 CSV
    out_top_csv = f"{OUTPUT_PREFIX}_PMI_top{TOP_N_PLOT}.csv"
    pmi_df.head(TOP_N_PLOT).to_csv(out_top_csv, index=False, encoding="utf-8-sig")
    print(f"Top {TOP_N_PLOT} PMI 结果已保存：{out_top_csv}")

    # 4. 可视化
    plot_top_pmi(pmi_df, topn=TOP_N_PLOT, out_prefix=f"{OUTPUT_PREFIX}_PMI_top{TOP_N_PLOT}")
    print(f"可视化图表已保存：{OUTPUT_PREFIX}_PMI_top{TOP_N_PLOT}.png")

    # 若启用字符级补充统计
    if CHAR_LEVEL:
        # 选取若干关注字符（可以手动列出，或从 pmi_df 前几获得）
        # 这里示例：从 PMI top 60 里抽出长度为1的字符候选
        cand = pmi_df.head(60)["word"].tolist()
        chars = sorted([w for w in cand if len(w) == 1])
        if chars:
            char_df = char_level_stats(token_series, TARGET, chars, mode=CHAR_MODE)
            char_df.to_excel(f"{OUTPUT_PREFIX}_char_level_{CHAR_MODE}.xlsx", index=False)
            print(f"字符级补充统计已保存：{OUTPUT_PREFIX}_char_level_{CHAR_MODE}.xlsx")
        else:
            print("未发现单字符候选用于字符级统计（PMI top 中无单字符）。")

    # 还可以把带 filtered tokens 的数据另存为新文件（保留原始分词）
    df.to_excel(f"{OUTPUT_PREFIX}_with_filtered_tokens.xlsx", index=False)
    print(f"包含过滤列的新表已保存：{OUTPUT_PREFIX}_with_filtered_tokens.xlsx")

    print("\n所有输出文件生成完成！")

if __name__ == "__main__":
    main()