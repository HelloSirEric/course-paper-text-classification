# modern_tfidf_analysis.py
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# ----------------- 可配置参数 -----------------
INPUT_FILE = "contemporary_corpus.xlsx"      # 包含列 '时间' 和 '分词结果'
TIME_COL = "时间"
TOKEN_COL = "分词结果"
STOPWORDS_FILE = "stopwords_cn.txt"
OUTPUT_XLSX = "modern_tfidf_results.xlsx"
OUT_PLOTS_DIR = "modern_tfidf_plots"
TOP_N = 8              # 每期显示前 TOP_N 词
GROUPING_MODE = "custom"  # "decade" 或 "custom"
# 若使用 custom 分期，则用下面的 bins 与 labels（左闭右开）
CUSTOM_BINS = [1900, 1907, 1911, 1927, 1949, 1960, 1966, 1968, 1976, 1989, 2001, 2016]
CUSTOM_LABELS = [
    "1900-1906","1907-1910","1911-1926","1927-1948","1949-1959",
    "1960-1965","1966-1967","1968-1975","1976-1988","1989-2000","2001-2015"
]
# -----------------------------------------------

# 字体设置（如需可替换为机器字体）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs(OUT_PLOTS_DIR, exist_ok=True)

# 读取停用词（转换为 list 供 sklearn 使用）
if not os.path.exists(STOPWORDS_FILE):
    raise FileNotFoundError(f"未找到停用词文件：{STOPWORDS_FILE}")
with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
    stopwords = [w.strip() for w in f if w.strip()]

# 读取语料
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"未找到输入文件：{INPUT_FILE}")
df = pd.read_excel(INPUT_FILE, engine="openpyxl")

if TIME_COL not in df.columns or TOKEN_COL not in df.columns:
    raise ValueError(f"输入文件必须包含列：'{TIME_COL}' 和 '{TOKEN_COL}'")

# 构建 period 列
def build_periods(df):
    if GROUPING_MODE == "decade":
        def decade_label(y):
            try:
                y = int(y)
                d = (y // 10) * 10
                return f"{d}s"
            except:
                return "unknown"
        df['period'] = df[TIME_COL].apply(decade_label)
    elif GROUPING_MODE == "custom":
        if not (len(CUSTOM_BINS) >= 2 and len(CUSTOM_LABELS) == len(CUSTOM_BINS)-1):
            raise ValueError("CUSTOM_BINS 与 CUSTOM_LABELS 长度不匹配")
        df['_year'] = pd.to_numeric(df[TIME_COL], errors='coerce')
        df = df.dropna(subset=['_year']).copy()
        df['period'] = pd.cut(df['_year'], bins=CUSTOM_BINS, labels=CUSTOM_LABELS, right=False)
        df = df.drop(columns=['_year'])
    else:
        raise ValueError("GROUPING_MODE 必须是 'decade' 或 'custom'")
    return df

df = build_periods(df)
df = df.dropna(subset=['period']).copy()

# 为每个 period 合并为一个大文档（空格分隔）
periods = list(df['period'].unique())
documents = []
period_order = []  # 保持顺序
for p in periods:
    sub = df[df['period'] == p]
    text = " ".join(sub[TOKEN_COL].astype(str).tolist())
    documents.append(text)
    period_order.append(str(p))

# 计算 TF-IDF（文档级）
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\S+\b", stop_words=stopwords)
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

# 保存每期所有词 TF-IDF 到 Excel（每期一个 sheet）
with pd.ExcelWriter(OUTPUT_XLSX, engine='openpyxl') as writer:
    for i, p in enumerate(period_order):
        row = tfidf_matrix[i].toarray()[0]
        scores = dict(zip(feature_names, row))
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        df_out = pd.DataFrame(sorted_items, columns=['词', 'TF-IDF'])
        sheet_name = str(p)[:31]
        df_out.to_excel(writer, sheet_name=sheet_name, index=False)
print(f"TF-IDF 结果已保存到 {OUTPUT_XLSX}")

# 每期画柱状图并保存（top N）
for i, p in enumerate(period_order):
    row = tfidf_matrix[i].toarray()[0]
    scores = dict(zip(feature_names, row))
    top_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    if not top_items:
        continue
    words, vals = zip(*top_items)
    plt.figure(figsize=(8,4))
    plt.bar(words, vals, color='skyblue')
    plt.title(f"{p} 期 TF-IDF 前{TOP_N} 词")
    plt.xticks(rotation=30, ha='right')
    plt.ylabel("TF-IDF")
    plt.tight_layout()
    fname = os.path.join(OUT_PLOTS_DIR, f"tfidf_{str(p)}_top{TOP_N}.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"已保存图像: {fname}")

print("全部完成。")
