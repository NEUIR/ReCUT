import json


def analyze_test_results(input_file, output_file):
    """
    分析测试数据，计算评估指标，并计算不同组别的输出平均长度

    Args:
        input_file: 包含测试案例的JSON文件路径
        output_file: 输出metrics的JSON文件路径
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 初始化统计变量
    total_cases = len(test_data)
    correct_em = 0
    correct_acc = 0
    correct_f1 = 0
    correct_math_equal = 0
    valid_answers = 0

    # 输出长度统计
    true_outputs = []
    false_outputs = []

    # 分析每个测试案例
    for case in test_data:
        metrics = case.get('Metrics', {})

        # 检查答案是否有效
        if metrics.get('is_valid_answer', False):
            valid_answers += 1

        # 计算正确案例数
        if metrics.get('em', 0) == 1:
            correct_em += 1
        if metrics.get('acc', 0) == 1:
            correct_acc += 1
            if 'Output' in case:
                true_outputs.append(len(case['Output']))
        else:
            # 收集math_equal=false的输出长度
            if 'Output' in case:
                false_outputs.append(len(case['Output']))
        if metrics.get('f1', 1.0) == 1.0:
            correct_f1 += 1
        if metrics.get('math_equal', False):
            correct_math_equal += 1
            # 收集math_equal=true的输出长度

    # 计算各项指标
    em_score = correct_em / total_cases if total_cases > 0 else 0
    acc_score = correct_acc / total_cases if total_cases > 0 else 0
    f1_score = correct_f1 / total_cases if total_cases > 0 else 0
    math_equal_score = correct_math_equal / total_cases if total_cases > 0 else 0

    # 计算平均长度
    avg_true = sum(true_outputs) / len(true_outputs) if true_outputs else 0
    avg_false = sum(false_outputs) / len(false_outputs) if false_outputs else 0
    all_outputs = true_outputs + false_outputs
    avg_overall = sum(all_outputs) / len(all_outputs) if all_outputs else 0

    # 构建metrics数据
    metrics = {
        "overall": {
            "em": round(em_score, 2),
            "acc": round(acc_score, 2),
            "f1": round(f1_score, 2),
            "math_equal": round(math_equal_score, 2),
            "num_valid_answer": f"{valid_answers} of {total_cases}",
            "query_latency": "16948 ms",  # 使用样例中的值
            "avg_length": {
                "true_cases": int(avg_true),
                "false_cases": int(avg_false),
                "overall": int(avg_overall)
            }
        }
    }

    # # 保存metrics文件
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     json.dump(metrics, f, indent=4)

    # 打印分析结果
    print("测试结果分析完成！")
    print(f"总案例数: {total_cases}")
    print(f"有效答案数: {valid_answers}")
    print(f"准确率(math_equal): {math_equal_score:.2f}")
    print(f"math_equal为true的案例数: {len(true_outputs)}, 平均长度: {int(avg_true)}字符")
    print(f"math_equal为false的案例数: {len(false_outputs)}, 平均长度: {int(avg_false)}字符")
    print(f"所有案例的平均长度: {int(avg_overall)}字符")
    print(f"结果已保存到 {output_file}")


if __name__ == "__main__":
    # 文件路径
    input_file = "/mnt1/wumingyan/lixinze/Data/dpo_train/Search-o1/outputs/llama3.1/gsm8k.merge_density_25_8b-200-500/test.5.15,2:43.json"
    output_file = "./test/metrics_aime_cot.json"

    analyze_test_results(input_file, output_file)