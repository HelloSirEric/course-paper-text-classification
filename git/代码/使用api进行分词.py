import pandas as pd
import requests
import json
import time
from tqdm import tqdm


def batch_process_excel(input_file, output_file, api_key, batch_delay=1):
    """批量处理Excel文件中的文本"""

    # 读取Excel文件
    df = pd.read_excel(input_file)
    print(f"成功读取文件，共{len(df)}行数据")

    # 准备结果列表
    segmented_results = [""] * len(df)
    costs = [0] * len(df)

    # 处理每一行
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理进度"):
        text = row['段落文本']

        # 跳过空文本
        if pd.isna(text) or str(text).strip() == "":
            segmented_results[idx] = ""
            costs[idx] = 0
            continue

        try:
            # 调用API分词
            result, cost = call_deepseek_v3(str(text), api_key)
            segmented_results[idx] = result
            costs[idx] = cost

            # 显示进度
            print(f"第{idx + 1}行处理成功，成本: ¥{cost:.4f}")

            # 延迟避免频率限制
            time.sleep(batch_delay)

        except Exception as e:
            print(f"第{idx + 1}行处理失败: {e}")
            segmented_results[idx] = ""
            costs[idx] = 0

    # 添加到DataFrame
    df['V3分词结果'] = segmented_results
    df['分词成本'] = costs

    # 保存结果
    df.to_excel(output_file, index=False)

    total_cost = sum(costs)
    print(f"\n处理完成！总成本: ¥{total_cost:.4f}")
    print(f"结果已保存到: {output_file}")

    return df


def call_deepseek_v3(text, api_key):
    """调用DeepSeek-V3进行分词"""
    url = "https://api.siliconflow.cn/v1/chat/completions"

    prompt = f"""
你是一个古汉语语言学专家。请对以下文言文文本进行专业分词，要求：
1. 保持文言文固定搭配完整
2. 专有名词不拆分  
3. 输出格式：仅返回用空格分隔的词语列表

文本：「{text}」
"""

    data = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.1
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content'].strip()

        # 计算成本（估算）
        input_tokens = len(text) / 3  # 粗略估算
        output_tokens = len(content) / 3
        cost = (input_tokens * 0.14 + output_tokens * 0.28) / 1000000

        return content, cost
    else:
        raise Exception(f"API调用失败: {response.status_code}")


# 使用示例
if __name__ == "__main__":
    api_key = "******"
    input_file = "geming_ancient.xlsx"
    output_file = "geming_ancient_v3_segmented.xlsx"

    result_df = batch_process_excel(input_file, output_file, api_key)