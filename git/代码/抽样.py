import pandas as pd
import random
from collections import Counter


def intelligent_sampling(input_file, sample_size=350, random_seed=42):
    """
    从完整数据中智能抽样，确保样本多样性
    """
    print("正在加载完整数据...")

    # 明确指定Excel引擎
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file, engine='openpyxl')

    print(f"原始数据总量: {len(df)} 条")
    print(f"典籍种类: {df['典籍名称'].nunique()} 种")

    # 设置随机种子确保可重复性
    random.seed(random_seed)

    # 策略：分层抽样确保多样性
    sampled_data = []

    # 查看典籍分布
    book_counts = df['典籍名称'].value_counts()
    print(f"\n典籍分布概况:")
    print(f"- 出现100次以上的典籍: {len(book_counts[book_counts >= 100])} 种")
    print(f"- 出现10-99次的典籍: {len(book_counts[(book_counts >= 10) & (book_counts < 100)])} 种")
    print(f"- 出现2-9次的典籍: {len(book_counts[(book_counts >= 2) & (book_counts < 10)])} 种")
    print(f"- 只出现1次的典籍: {len(book_counts[book_counts == 1])} 种")

    # 分层抽样策略
    # 1. 高频典籍：抽样但不过度集中
    major_books = book_counts[book_counts >= 10].index.tolist()
    print(f"\n从 {len(major_books)} 种高频典籍中抽样...")

    for book in major_books:
        book_data = df[df['典籍名称'] == book]
        # 高频典籍抽样比例较低，避免过度集中
        sample_n = max(1, min(3, len(book_data) // 20))
        sampled = book_data.sample(n=sample_n, random_state=random_seed)
        sampled_data.append(sampled)

    # 2. 中频典籍：适度抽样
    medium_books = book_counts[(book_counts >= 2) & (book_counts < 10)].index.tolist()
    print(f"从 {len(medium_books)} 种中频典籍中抽样...")

    for book in medium_books:
        book_data = df[df['典籍名称'] == book]
        sample_n = min(2, len(book_data))
        sampled = book_data.sample(n=sample_n, random_state=random_seed)
        sampled_data.append(sampled)

    # 3. 低频典籍：尽量多覆盖
    rare_books = book_counts[book_counts == 1].index.tolist()
    print(f"从 {len(rare_books)} 种低频典籍中抽样...")

    # 随机选择一部分低频典籍
    rare_sample = random.sample(rare_books, min(100, len(rare_books)))
    for book in rare_sample:
        book_data = df[df['典籍名称'] == book]
        sampled_data.append(book_data)

    # 合并所有样本
    if sampled_data:  # 确保列表不为空
        final_sample = pd.concat(sampled_data, ignore_index=True)
    else:
        final_sample = df.sample(n=sample_size, random_state=random_seed)

    # 如果总数超过目标，随机削减到目标数量
    if len(final_sample) > sample_size:
        final_sample = final_sample.sample(n=sample_size, random_state=random_seed)
    elif len(final_sample) < sample_size:
        # 如果不足，从剩余数据中补充
        selected_indices = set(final_sample.index)
        remaining_data = df[~df.index.isin(selected_indices)]
        additional_needed = sample_size - len(final_sample)
        if len(remaining_data) >= additional_needed:
            additional_samples = remaining_data.sample(n=additional_needed, random_state=random_seed)
            final_sample = pd.concat([final_sample, additional_samples], ignore_index=True)
        else:
            # 如果剩余数据不足，全部选取
            final_sample = pd.concat([final_sample, remaining_data], ignore_index=True)

    print(f"\n=== 抽样结果 ===")
    print(f"最终样本量: {len(final_sample)} 条")
    print(f"覆盖典籍种类: {final_sample['典籍名称'].nunique()} 种")

    # 显示样本分布
    print("\n样本中典籍分布:")
    book_dist = final_sample['典籍名称'].value_counts()
    print("出现3次以上的典籍:")
    for book, count in book_dist[book_dist >= 3].items():
        print(f"  {book}: {count}条")

    return final_sample


def save_sampled_data(sample_df, output_file):
    """保存抽样结果"""
    if output_file.endswith('.xlsx'):
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            sample_df.to_excel(writer, index=False, sheet_name='抽样语料')
    else:
        sample_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n抽样数据已保存到: {output_file}")


def main():
    # 文件路径配置
    input_file = "BCC_革命语料_完整版.xlsx"  # 代码1的输出
    output_file = "BCC_革命语料_抽样版.xlsx"
    sample_size = 800

    # 执行智能抽样
    sampled_df = intelligent_sampling(input_file, sample_size)

    # 保存结果
    save_sampled_data(sampled_df, output_file)

    # 显示抽样示例
    print("\n=== 抽样数据示例 ===")
    for i, (_, row) in enumerate(sampled_df.head(5).iterrows()):
        print(f"\n示例 {i + 1}:")
        print(f"典籍: {row['典籍名称']}")
        print(f"正文: {row['原始正文'][:80]}...")


if __name__ == "__main__":
    main()