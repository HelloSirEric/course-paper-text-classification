import pandas as pd
import re
import os


def parse_bcc_corpus(file_path):
    """
    解析BCC语料库文件，提取典籍名称和正文
    """
    print("正在读取BCC语料文件...")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 使用正则表达式匹配 <B>标签和正文
    pattern = r'<B>([^<]+)</B>([^<]*(?:<U>革命</U>[^<]*)?)'
    matches = re.findall(pattern, content)

    print(f"找到 {len(matches)} 个条目")

    # 提取数据
    data = []
    for i, (book_info, text) in enumerate(matches):
        # 提取典籍名称（第一个词）
        book_name = book_info.split(' ')[0].strip()

        # 清洗正文：移除<U>标签但保留内容
        clean_text = re.sub(r'<U>|</U>', '', text).strip()

        # 只保留包含"革命"的条目
        if '革命' in clean_text:
            data.append({
                '序号': i + 1,
                '典籍名称': book_name,
                '原始正文': clean_text,
                '分类路径': book_info  # 保留完整分类信息用于后续分析
            })

    print(f"其中包含'革命'的条目: {len(data)} 个")
    return data


def save_full_data(data, output_file):
    """保存完整数据到CSV"""
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"完整数据已保存到: {output_file}")
    print(f"总条目数: {len(df)}")

    # 显示数据统计
    print("\n=== 数据统计 ===")
    print(f"唯一典籍数量: {df['典籍名称'].nunique()}")
    print("\n前10个高频典籍:")
    print(df['典籍名称'].value_counts().head(10))


def main():
    # 文件路径配置
    input_file = "bcc_corpus.txt"  
    output_file = "BCC_革命语料_完整版.xlsx"

    if not os.path.exists(input_file):
        print(f"错误：文件 {input_file} 不存在")
        return

    # 解析数据
    data = parse_bcc_corpus(input_file)

    if not data:
        print("未找到包含'革命'的条目")
        return

    # 保存完整数据
    save_full_data(data, output_file)

    # 显示示例
    print("\n=== 数据示例 ===")
    sample_df = pd.DataFrame(data[:3])
    for _, row in sample_df.iterrows():
        print(f"\n典籍: {row['典籍名称']}")
        print(f"正文: {row['原始正文'][:80]}...")


if __name__ == "__main__":
    main()