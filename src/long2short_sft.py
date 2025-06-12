import json
import re

def process_math_problems(input_file, output_file="processed_solutions.json"):
    """
    从JSON文件中提取数学问题数据并处理

    参数:
    input_file: 输入文件路径
    output_file: 输出文件路径
    """
    try:
        # 读取文件内容
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 如果内容在document_content标签中，提取它
        doc_match = re.search(r'<document_content>\s*(.*?)\s*</document_content>', content, re.DOTALL)
        if doc_match:
            content = doc_match.group(1).strip()

        # 尝试解析JSON
        try:
            # 先尝试直接解析
            data = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"直接解析JSON失败: {str(e)}")

            # 如果失败，尝试包装成JSON数组
            if not content.strip().startswith('['):
                content = '[' + content
            if not content.strip().endswith(']'):
                content = content + ']'

            # 清理常见问题
            content = content.replace(',]', ']')  # 删除尾随逗号

            try:
                data = json.loads(content)
                print(f"包装为数组后成功解析，找到 {len(data)} 个问题")
            except json.JSONDecodeError as e:
                print(f"包装后解析仍然失败: {str(e)}")

                # 尝试单独解析每个问题
                pattern = r'\{\s*"problem".*?"is_correct":\s*(?:true|false)\s*\}'
                matches = re.findall(pattern, content, re.DOTALL)

                if not matches:
                    raise ValueError("无法从文件中识别任何有效的问题")

                data = []
                for match_text in matches:
                    try:
                        obj = json.loads(match_text)
                        data.append(obj)
                    except json.JSONDecodeError:
                        print(f"解析问题对象失败")

        print(f"成功读取JSON数据，找到 {len(data)} 个数学问题")

        results = []

        # 处理每个题目
        for item in data:
            problem = item.get("problem", "")
            final_answer = item.get("final_answer", "")
            selected_steps = item.get("selected_steps", [])

            # 构建新的solution
            new_solution = ""

            # 直接拼接每个步骤，不加标记
            for step in selected_steps:
                new_solution += step.lstrip() + "\n\n"

            # 去除末尾多余的换行
            new_solution = new_solution.rstrip()

            # 保存处理结果
            results.append({
                "problem": problem,
                "solution": new_solution,
                "final_answer": final_answer
            })

        # 保存结果到JSON文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"处理完成，结果已保存到: {output_file}")
        print(f"成功处理了 {len(results)} 个数学问题")
        return True

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        return False

# 使用示例
if __name__ == "__main__":
    input_file = "/data1/jiyifan/cot_pro/instruct_reward_data/optimal_solutions_4k_correct.json"  # 你的JSON文件路径
    output_file = "/data1/jiyifan/cot_pro/instruct_reward_data/sft_data.json"
    process_math_problems(input_file, output_file)
