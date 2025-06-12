#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试结果分析工具
分析测试数据，计算评估指标，并计算不同组别的输出平均长度
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime


def analyze_test_results(input_file, output_file=None, save_output=True):
    """
    分析测试数据，计算评估指标，并计算不同组别的输出平均长度

    Args:
        input_file: 包含测试案例的JSON文件路径
        output_file: 输出metrics的JSON文件路径（可选）
        save_output: 是否保存输出文件

    Returns:
        dict: 分析结果指标
    """
    try:
        # 读取输入文件
        print(f"正在读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        print(f"成功加载 {len(test_data)} 个测试案例")

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
        for idx, case in enumerate(test_data):
            metrics = case.get('Metrics', {})

            # 检查答案是否有效
            if metrics.get('is_valid_answer', False):
                valid_answers += 1

            # 计算正确案例数
            if metrics.get('em', 0) == 1:
                correct_em += 1
            if metrics.get('acc', 0) == 1:
                correct_acc += 1
            if metrics.get('f1', 0) == 1.0:
                correct_f1 += 1

            # 处理 math_equal 指标
            if metrics.get('math_equal', False):
                correct_math_equal += 1
                # 收集math_equal=true的输出长度
                if 'Output' in case:
                    true_outputs.append(len(case['Output']))
            else:
                # 收集math_equal=false的输出长度
                if 'Output' in case:
                    false_outputs.append(len(case['Output']))

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
        metrics_result = {
            "overall": {
                "em": round(em_score, 4),
                "acc": round(acc_score, 4),
                "f1": round(f1_score, 4),
                "math_equal": round(math_equal_score, 4),
                "num_valid_answer": f"{valid_answers} of {total_cases}",
                "avg_length": {
                    "true_cases": int(avg_true),
                    "false_cases": int(avg_false),
                    "overall": int(avg_overall)
                }
            },
            "statistics": {
                "total_cases": total_cases,
                "correct_em": correct_em,
                "correct_acc": correct_acc,
                "correct_f1": correct_f1,
                "correct_math_equal": correct_math_equal,
                "valid_answers": valid_answers,
                "true_cases_count": len(true_outputs),
                "false_cases_count": len(false_outputs)
            },
            "metadata": {
                "input_file": str(input_file),
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }

        # 保存metrics文件
        if save_output and output_file:
            # 确保输出目录存在
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_result, f, indent=4, ensure_ascii=False)
            print(f"\n✅ 结果已保存到: {output_file}")

        # 打印分析结果
        print("\n" + "=" * 50)
        print("📊 测试结果分析报告")
        print("=" * 50)
        print(f"输入文件: {input_file}")
        print(f"\n📈 整体指标:")
        print(f"  - 总案例数: {total_cases}")
        print(f"  - 有效答案数: {valid_answers}")
        print(f"  - EM得分: {em_score:.2%} ({correct_em}/{total_cases})")
        print(f"  - ACC得分: {acc_score:.2%} ({correct_acc}/{total_cases})")
        print(f"  - F1得分: {f1_score:.2%} ({correct_f1}/{total_cases})")
        print(f"  - Math Equal得分: {math_equal_score:.2%} ({correct_math_equal}/{total_cases})")

        print(f"\n📏 输出长度分析:")
        print(f"  - Math Equal = True 的案例:")
        print(f"    • 数量: {len(true_outputs)}")
        print(f"    • 平均长度: {int(avg_true)} 字符")
        print(f"  - Math Equal = False 的案例:")
        print(f"    • 数量: {len(false_outputs)}")
        print(f"    • 平均长度: {int(avg_false)} 字符")
        print(f"  - 所有案例平均长度: {int(avg_overall)} 字符")

        if len(true_outputs) > 0 and len(false_outputs) > 0:
            length_diff = avg_true - avg_false
            print(f"  - 长度差异: {abs(int(length_diff))} 字符 ({'True更长' if length_diff > 0 else 'False更长'})")

        print("=" * 50)

        return metrics_result

    except FileNotFoundError:
        print(f"❌ 错误: 找不到输入文件 {input_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ 错误: 解析JSON文件失败 - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        sys.exit(1)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='分析测试结果，计算评估指标和输出长度统计',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 分析并保存结果
  %(prog)s --input test_results.json --output metrics.json

  # 只分析不保存（仅显示结果）
  %(prog)s --input test_results.json --no-save

  # 使用短参数
  %(prog)s -i test.json -o metrics.json

输出格式:
  程序会计算以下指标:
  - EM (Exact Match) 得分
  - ACC (Accuracy) 得分  
  - F1 得分
  - Math Equal 得分
  - 不同组别的输出平均长度
        """
    )

    # 必需参数
    parser.add_argument('-i', '--input',
                        required=True,
                        help='输入的测试结果JSON文件路径')

    # 可选参数
    parser.add_argument('-o', '--output',
                        help='输出的metrics JSON文件路径（可选）')

    parser.add_argument('--no-save',
                        action='store_true',
                        help='不保存输出文件，仅显示分析结果')

    # 其他选项
    parser.add_argument('-v', '--version',
                        action='version',
                        version='%(prog)s 1.0')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 确定是否保存输出
    save_output = not args.no_save

    # 如果需要保存但没有指定输出文件，自动生成
    output_file = args.output
    if save_output and not output_file:
        input_path = Path(args.input)
        output_file = str(input_path.parent / f"{input_path.stem}_metrics.json")
        print(f"自动生成输出文件名: {output_file}")

    # 执行分析
    analyze_test_results(args.input, output_file, save_output)

    print("\n✅ 分析完成！")


if __name__ == "__main__":
    main()