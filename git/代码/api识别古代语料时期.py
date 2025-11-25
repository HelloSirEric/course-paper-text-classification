import pandas as pd
import requests
import json
import time
from tqdm import tqdm
import os


def identify_dynasty_batch(input_file, output_file, api_key, batch_delay=1):
    """批量识别典籍的大时期"""

    # 读取抽样数据
    df = pd.read_excel(input_file)
    print(f"成功读取文件，共{len(df)}行数据")

    # 准备结果列表
    period_results = [""] * len(df)
    period_reasons = [""] * len(df)
    costs = [0] * len(df)

    # 处理每一行
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="大时期识别进度"):
        book_name = row['典籍名称']

        # 跳过空典籍名
        if pd.isna(book_name) or str(book_name).strip() == "":
            period_results[idx] = "未知"
            period_reasons[idx] = "典籍名称为空"
            costs[idx] = 0
            continue

        try:
            # 调用API识别大时期
            period, reason, cost = call_period_identification(str(book_name), api_key)
            period_results[idx] = period
            period_reasons[idx] = reason
            costs[idx] = cost

            # 显示进度
            if (idx + 1) % 10 == 0:
                print(f"已处理 {idx + 1}/{len(df)}: 《{book_name}》→ {period}")

            # 延迟避免频率限制
            time.sleep(batch_delay)

        except Exception as e:
            print(f"第{idx + 1}行处理失败: {e}")
            period_results[idx] = "识别失败"
            period_reasons[idx] = str(e)
            costs[idx] = 0

    # 添加到DataFrame
    df['大时期'] = period_results
    df['识别依据'] = period_reasons
    df['识别成本'] = costs

    # 保存结果
    df.to_excel(output_file, index=False)

    total_cost = sum(costs)
    print(f"\n大时期识别完成！")
    print(f"总成本: ¥{total_cost:.4f}")
    print(f"时期分布: {df['大时期'].value_counts().to_dict()}")
    print(f"结果已保存到: {output_file}")

    return df


def call_period_identification(book_name, api_key):
    """调用API识别典籍大时期"""
    url = "https://api.siliconflow.cn/v1/chat/completions"

    prompt = f"""
请判断典籍《{book_name}》的成书时期，并归类到以下大时期之一：

【大时期分类】
1. 先秦（夏商周、春秋战国）
2. 两汉（西汉、东汉）  
3. 魏晋南北朝
4. 隋唐五代
5. 宋辽金
6. 元代
7. 明代
8. 清代
9. 近代（1840-1919）
10. 现代（1919年以后）

要求：
1. 直接回复大时期名称，如：先秦、两汉、魏晋南北朝、隋唐五代等
2. 如果无法确定确切时期，请给出最可能的大时期
3. 只需回复大时期名称，不要添加其他文字

典籍名称：《{book_name}》
"""

    data = {
        "model": "deepseek-ai/DeepSeek-V3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0.1
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 修复编码问题：明确指定编码
    try:
        # 方法1：使用json参数自动处理编码
        response = requests.post(
            url,
            headers=headers,
            json=data,  # 使用json参数而不是data=json.dumps(data)
            timeout=30
        )
    except requests.exceptions.RequestException as e:
        # 方法2：如果方法1失败，尝试手动编码
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(data, ensure_ascii=False).encode('utf-8'),  # 明确使用UTF-8编码
                timeout=30
            )
        except Exception as e2:
            raise Exception(f"请求失败: {e2}")

    if response.status_code == 200:
        result = response.json()
        period = result['choices'][0]['message']['content'].strip()

        # 清理回复内容，只保留大时期
        period = clean_period_response(period)

        # 计算成本（估算）
        input_tokens = len(prompt) / 3
        output_tokens = len(period) / 3
        cost = (input_tokens * 0.14 + output_tokens * 0.28) / 1000000

        return period, "自动识别", cost
    else:
        raise Exception(f"API调用失败: {response.status_code}, 响应: {response.text}")


def clean_period_response(response):
    """清理大时期识别结果"""
    # 定义标准大时期列表
    standard_periods = [
        '先秦', '两汉', '魏晋南北朝', '隋唐五代',
        '宋辽金', '元代', '明代', '清代'
    ]

    # 移除可能的说明文字，只保留大时期名称
    for period in standard_periods:
        if period in response:
            return period

    # 处理一些常见变体
    period_mapping = {
        '汉': '两汉',
        '唐': '隋唐五代',
        '宋': '宋辽金',
        '元': '元代',
        '明': '明代',
        '清': '清代',

    }

    for variant, standard in period_mapping.items():
        if variant in response:
            return standard

    # 如果找不到明确大时期，返回原响应（清理后）
    cleaned = response.replace('时期：', '').replace('时代：', '').strip()
    return cleaned if cleaned else "未知"


# 使用示例
if __name__ == "__main__":
    api_key = "******"

    input_file = "BCC_革命语料_抽样版.xlsx"  # 800条抽样数据
    output_file = "BCC_革命语料_带大时期.xlsx"

    result_df = identify_dynasty_batch(input_file, output_file, api_key)