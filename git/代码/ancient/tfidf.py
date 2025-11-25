import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# ---------- 参数设置 ----------
STOPWORDS_FILE = "stopwords_cn.txt"
INPUT_FILE = "ancient_corpus.xlsx"
OUTPUT_FILE = "tfidf_results.xlsx"
TOP_N = 5  # 每个时期显示前 top N 词

# ---------- 读取停用词 ----------
with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f if line.strip()]  # 转为 list

# ---------- 读取语料 ----------
df = pd.read_excel(INPUT_FILE)  # 列：序号,典籍名称,原始正文,大时期,分词文本

# ---------- 构建文档：每个大时期当作一个文档 ----------
documents = []
periods = []
for period in df['大时期'].unique():
    period_df = df[df['大时期'] == period]
    # 拼接所有分词文本，用空格分隔
    text = " ".join(period_df['分词文本'].tolist())
    documents.append(text)
    periods.append(period)

# ---------- TF-IDF 计算 ----------
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

# ---------- 输出 Excel ----------
with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
    for i, period in enumerate(periods):
        row = tfidf_matrix[i].toarray()[0]
        tfidf_scores = dict(zip(feature_names, row))
        sorted_tfidf = dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True))
        temp_df = pd.DataFrame(list(sorted_tfidf.items()), columns=['词', 'TF-IDF'])
        temp_df.to_excel(writer, sheet_name=period, index=False)

print(f"TF-IDF 结果已保存到 {OUTPUT_FILE}")

# ---------- 可视化历时 top N 词 ----------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False

for i, period in enumerate(periods):
    row = tfidf_matrix[i].toarray()[0]
    tfidf_scores = dict(zip(feature_names, row))
    sorted_items = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
    top_words, top_values = zip(*sorted_items)

    plt.figure(figsize=(8, 4))
    plt.bar(top_words, top_values, color='skyblue')
    plt.title(f"{period}时期 TF-IDF 前{TOP_N}词")
    plt.ylabel("TF-IDF")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
