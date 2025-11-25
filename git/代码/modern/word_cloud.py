import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os

# ---------- 参数设置 ----------
INPUT_FILE = "contemporary_corpus.xlsx"        # 近现代语料
STOPWORDS_FILE = "stopwords_cn.txt"
OUTPUT_TFIDF_FILE = "modern_tfidf.xlsx"
HEATMAP_FILE = "modern_tfidf_heatmap.png"
WORDCLOUD_DIR = "modern_wordclouds/"
TOP_N = 10  # 每个时期保留 top N 词

os.makedirs(WORDCLOUD_DIR, exist_ok=True)

# ---------- 中文字体 ----------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 读取停用词 ----------
with open(STOPWORDS_FILE, 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f if line.strip()]

# ---------- 读取语料 ----------
df = pd.read_excel(INPUT_FILE)  # 列：序号, 时间, 正文, 分词结果
df['时间'] = df['时间'].astype(int)

# ---------- 分期策略 ----------
# 这里示例按十年分期
df['period'] = (df['时间'] // 10) * 10

documents = []
periods = []
for period in sorted(df['period'].unique()):
    period_df = df[df['period'] == period]
    text = " ".join(period_df['分词结果'].tolist())
    documents.append(text)
    periods.append(str(period))

# ---------- TF-IDF ----------
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

# ---------- 构建热力图 ----------
heatmap_data = pd.DataFrame()
for i, period in enumerate(periods):
    row = tfidf_matrix[i].toarray()[0]
    tfidf_scores = dict(zip(feature_names, row))
    top_items = dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_N])
    temp_df = pd.DataFrame.from_dict(top_items, orient='index', columns=[period])
    heatmap_data = pd.concat([heatmap_data, temp_df], axis=1)

heatmap_data = heatmap_data.fillna(0).T  # 行：时期，列：词

plt.figure(figsize=(20, 10))
sns.heatmap(heatmap_data, cmap="YlOrRd", cbar=True, annot=False)  # 不显示数字
plt.title(f"近现代 TF-IDF 热力图（Top {TOP_N} 词）", fontsize=16)
plt.ylabel("时期")
plt.xlabel("词")
plt.tight_layout()
plt.savefig(HEATMAP_FILE)
plt.show()
print(f"历时热力图已保存到 {HEATMAP_FILE}")

# ---------- 生成词云 ----------
for i, period in enumerate(periods):
    row = tfidf_matrix[i].toarray()[0]
    tfidf_scores = dict(zip(feature_names, row))
    wc = WordCloud(font_path="simhei.ttf", background_color="white", width=800, height=400)
    wc.generate_from_frequencies(tfidf_scores)
    plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"{period}时期词云", fontsize=16)
    save_path = os.path.join(WORDCLOUD_DIR, f"{period}_wordcloud.png")
    plt.savefig(save_path)
    plt.show()
    print(f"{period} 词云已保存到 {save_path}")
