import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 参数设置 ----------
STOPWORDS_FILE = "stopwords_cn.txt"
INPUT_FILE = "ancient_corpus.xlsx"
OUTPUT_TFIDF_FILE = "tfidf_results.xlsx"
TOP_N = 8  # 每个时期展示 top N 词
HEATMAP_FILE = "tfidf_heatmap.png"
WORDCLOUD_DIR = "wordclouds/"

import os
os.makedirs(WORDCLOUD_DIR, exist_ok=True)

# ---------- 读取停用词 ----------
with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f if line.strip()]

# ---------- 读取语料 ----------
df = pd.read_excel(INPUT_FILE)  # 列：序号,典籍名称,原始正文,大时期,分词文本

# ---------- 构建文档：每个大时期当作一个文档 ----------
documents = []
periods = []
for period in df['大时期'].unique():
    period_df = df[df['大时期'] == period]
    text = " ".join(period_df['分词文本'].tolist())
    documents.append(text)
    periods.append(period)

# ---------- TF-IDF 计算 ----------
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)
tfidf_matrix = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

# ---------- 输出 Excel ----------
with pd.ExcelWriter(OUTPUT_TFIDF_FILE, engine='openpyxl') as writer:
    for i, period in enumerate(periods):
        row = tfidf_matrix[i].toarray()[0]
        tfidf_scores = dict(zip(feature_names, row))
        sorted_tfidf = dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True))
        temp_df = pd.DataFrame(list(sorted_tfidf.items()), columns=['词', 'TF-IDF'])
        temp_df.to_excel(writer, sheet_name=period, index=False)

print(f"TF-IDF 结果已保存到 {OUTPUT_TFIDF_FILE}")

# ---------- 构建历时热力图数据 ----------
heatmap_data = pd.DataFrame()
for i, period in enumerate(periods):
    row = tfidf_matrix[i].toarray()[0]
    tfidf_scores = dict(zip(feature_names, row))
    # 取 top N 词
    top_items = dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N])
    temp_df = pd.DataFrame.from_dict(top_items, orient='index', columns=[period])
    heatmap_data = pd.concat([heatmap_data, temp_df], axis=1)

heatmap_data = heatmap_data.fillna(0)  # 没有的词填0
heatmap_data = heatmap_data.T  # 行：时期，列：词

# ---------- 绘制热力图 ----------
plt.figure(figsize=(36,12))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd")
plt.title(f"各时期 TF-IDF 热力图（Top {TOP_N} 词）", fontsize=10)
plt.ylabel("时期")
plt.xlabel("词")
plt.tight_layout()
plt.savefig(HEATMAP_FILE)
plt.show()
print(f"历时热力图已保存到 {HEATMAP_FILE}")


# ---------- 每个大时期词云 ----------
for i, period in enumerate(periods):
    row = tfidf_matrix[i].toarray()[0]
    tfidf_scores = dict(zip(feature_names, row))
    wc = WordCloud(font_path="simhei.ttf", background_color="white", width=800, height=400)
    wc.generate_from_frequencies(tfidf_scores)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{period}时期词云", fontsize=16)
    save_path = os.path.join(WORDCLOUD_DIR, f"{period}_wordcloud.png")
    plt.savefig(save_path)
    plt.show()
    print(f"{period} 词云已保存到 {save_path}")
