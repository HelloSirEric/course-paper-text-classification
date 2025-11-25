# modern_pmi_analysis.py
"""
近现代语料 PMI 分析脚本（按时期/十年切片）
输入文件：modern_corpus.xlsx，须包含列 "时间"（年份整数） 和 "分词结果"（空格分词）
停用词：stopwords_cn.txt（每行一个词/标点）
输出：modern_out_* 文件（Excel / CSV / PNG）
"""

import os
import math
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# ----------------------------- 配置区（请按需修改） -----------------------------
INPUT_XLSX = "contemporary_corpus.xlsx"     # 输入文件（含 '时间' 和 '分词结果' 列）
TOKEN_COL = "分词结果"                # 分词列（已空格分词）
YEAR_COL = "时间"                     # 年份列（整数或可以转换为整数）
STOPWORDS_FILE = "stopwords_cn.txt"   # 停用词表（每行一个）
OUTPUT_PREFIX = "modern_out"          # 输出前缀
TARGET = "革命"                       # 中心词
MIN_COOC_FREQ = 5                     # 共现最小频次阈值（近现代可设高一些）
REMOVE_PUNCT = True                   # 是否把标点当停用词
TOP_N_PLOT = 20                       # 可视化取前 N
GROUPING_MODE = "custom"              # "decade" or "custom"
# 若 GROUPING_MODE == "custom"，请在下面定义分割点和标签（左闭右开）
# bins 是年份边界列表；labels 是每个区间的标签，len(labels) == len(bins)-1
CUSTOM_BINS = [1900, 1907, 1911, 1927, 1949, 1960, 1966, 1968, 1976, 1989, 2001, 2016]
CUSTOM_LABELS = [
    "1900-1906","1907-1910","1911-1926","1927-1948","1949-1959",
    "1960-1965","1966-1967","1968-1975","1976-1988","1989-2000","2001-2015"
]
# ---------------------------------------------------------------------------

# 设置中文字体，避免图表乱码（按需调整）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_stopwords(path, remove_punct=True):
    s = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                w = line.strip()
                if w:
                    s.add(w)
    if remove_punct:
        puncts = {"，", "。", "、", "；", "：", "？", "！", ",", ".", ":", ";",
                  "(", ")", "（", "）", "“", "”", "\"", "'", "‘", "’", "—", "–", "…", "·", "【", "】"}
        s |= puncts
    return s

def filter_tokens_str(token_str, stopwords):
    toks = [t for t in str(token_str).split() if t and t not in stopwords]
    return " ".join(toks)

def build_period_index(df):
    """
    根据 GROUPING_MODE 构建 df['period'] 列
    """
    if GROUPING_MODE == "decade":
        # 以 decade 为单位，例如 1900-1909 -> '1900s'
        def decade_label(y):
            try:
                y = int(y)
                d = (y // 10) * 10
                return f"{d}s"
            except:
                return "unknown"
        df['period'] = df[YEAR_COL].apply(decade_label)
    elif GROUPING_MODE == "custom":
        # 使用 CUSTOM_BINS 与 CUSTOM_LABELS
        if not (len(CUSTOM_BINS) >= 2 and len(CUSTOM_LABELS) == len(CUSTOM_BINS)-1):
            raise ValueError("CUSTOM_BINS 与 CUSTOM_LABELS 长度不匹配")
        # pd.cut requires numeric; coerce errors to NaN and drop later
        df['_year_numeric'] = pd.to_numeric(df[YEAR_COL], errors='coerce')
        df = df.dropna(subset=['_year_numeric']).copy()
        df['period'] = pd.cut(df['_year_numeric'], bins=CUSTOM_BINS, labels=CUSTOM_LABELS, right=False)
        df = df.drop(columns=['_year_numeric'])
    else:
        raise ValueError("GROUPING_MODE must be 'decade' or 'custom'")
    return df

def compute_counts_per_period(df_period, token_col, stopwords, target, min_cooc=1):
    """
    对一个 period 的句子（已过滤停用词）执行句子级统计：
    返回：N_sentences, count_target, count_word (句级出现次数), count_co (目标与词共现句数)
    """
    token_series = df_period[token_col].astype(str).tolist()
    N = 0
    count_target = 0
    count_word = Counter()
    count_co = Counter()
    for s in token_series:
        N += 1
        tokens = [t for t in s.split() if t and t not in stopwords]
        token_set = set(tokens)
        for w in token_set:
            count_word[w] += 1
        if target in token_set:
            count_target += 1
            for w in token_set:
                if w != target:
                    count_co[w] += 1
    return N, count_target, count_word, count_co

def compute_pmi_table(N, count_target, count_word, count_co, min_freq=1, smooth=1.0):
    rows = []
    for w, co in count_co.items():
        if co < min_freq:
            continue
        wcount = count_word.get(w, 0)
        numerator = (co * N) + smooth
        denominator = (count_target + smooth) * (wcount + smooth)
        pmi = math.log2(numerator / denominator) if denominator > 0 else float("-inf")
        rows.append((w, co, wcount, pmi))
    dfp = pd.DataFrame(rows, columns=["word", "cooc_count", "word_count", "pmi"])
    dfp = dfp.sort_values("pmi", ascending=False).reset_index(drop=True)
    return dfp

def plot_period_top(pmi_df, period_label, topn=20, out_prefix="modern_pmi"):
    df = pmi_df.head(topn)
    if df.empty:
        return
    plt.figure(figsize=(8, max(3, 0.25 * len(df))))
    plt.barh(df['word'][::-1], df['pmi'][::-1])
    plt.xlabel("PMI (log2)")
    plt.title(f"{period_label} — Top {topn} PMI with '{TARGET}'")
    plt.tight_layout()
    fname = f"{out_prefix}_{period_label.replace(' ', '_')}_top{topn}.png"
    plt.savefig(fname, dpi=200)
    plt.close()
    return fname

def main():
    # 检查输入
    if not os.path.exists(INPUT_XLSX):
        raise FileNotFoundError(f"未找到输入文件：{INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX, engine="openpyxl")
    if YEAR_COL not in df.columns or TOKEN_COL not in df.columns:
        raise ValueError(f"输入文件必须包含列：'{YEAR_COL}' 和 '{TOKEN_COL}'")

    # 加载停用词
    stopwords = load_stopwords(STOPWORDS_FILE, remove_punct=REMOVE_PUNCT)

    # 先生成过滤列，不覆盖原列
    df['tokens_no_stop'] = df[TOKEN_COL].astype(str).apply(lambda s: filter_tokens_str(s, stopwords))

    # 构建 period 列
    df = build_period_index(df)

    # 如果 custom 分期模式，有可能出现 NaN period（年份不在区间），过滤掉
    df = df.dropna(subset=['period']).copy()

    periods = sorted(df['period'].unique(), key=lambda x: str(x))

    # 输出工作目录
    out_dir = f"{OUTPUT_PREFIX}_pmi_outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 汇总每个 period 的 PMI 表，写入 Excel 的不同 sheet
    excel_path = os.path.join(out_dir, f"{OUTPUT_PREFIX}_PMI_by_period.xlsx")
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for per in periods:
            sub = df[df['period'] == per]
            N, count_target, count_word, count_co = compute_counts_per_period(sub, 'tokens_no_stop', stopwords, TARGET, min_cooc=MIN_COOC_FREQ)
            print(f"[INFO] Period={per} 句子数={N} | 含'{TARGET}'句子数={count_target} | 候选共现词数={len(count_co)}")
            if count_target == 0:
                print(f"  -> Period {per} 中未发现目标词，跳过。")
                continue
            pmi_df = compute_pmi_table(N, count_target, count_word, count_co, min_freq=MIN_COOC_FREQ, smooth=1.0)
            # 写 sheet
            sheet_name = str(per)[:31]  # Excel sheet 名称限制 31 字符
            pmi_df.to_excel(writer, sheet_name=sheet_name, index=False)
            # 保存每期的 top 可视化图
            img = plot_period_top(pmi_df, str(per), topn=TOP_N_PLOT, out_prefix=os.path.join(out_dir, f"{OUTPUT_PREFIX}"))
            if img:
                print(f"  -> 已保存图像: {img}")
    print(f"[DONE] 所有分期 PMI 已写入：{excel_path}")
    print(f"[DONE] 可视化文件保存在：{out_dir}")

if __name__ == "__main__":
    main()
