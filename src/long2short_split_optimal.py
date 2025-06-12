#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON文件拆分工具
根据is_correct字段将JSON文件拆分为正确和错误两个文件
"""

import json
import argparse
import sys
from pathlib import Path


def split_by_correctness(input_file, output_correct, output_incorrect):
    """
    根据is_correct值将JSON文件中的项目拆分为两个文件

    Args:
        input_file: 输入文件路径
        output_correct: 保存正确项的文件路径
        output_incorrect: 保存错误项的文件路径
    """
    try:
        # 读取输入文件
        print(f"正在读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 查找项目列表 (solutions, results等)
        items_list = None

        # 优先查找常见的键名
        common_keys = ["solutions", "results", "items", "data"]
        for key in common_keys:
            if key in data and isinstance(data[key], list):
                items_list = data[key]
                print(f"在 '{key}' 字段中找到项目列表")
                break

        # 如果没找到，查找第一个列表类型的值
        if items_list is None:
            for key, value in data.items():
                if isinstance(value, list):
                    items_list = value
                    print(f"在 '{key}' 字段中找到项目列表")
                    break

        if not items_list:
            print(f"错误: 在 {input_file} 中找不到项目列表")
            return False

        print(f"找到 {len(items_list)} 个项目")

        # 根据is_correct拆分项目
        correct_items = []
        incorrect_items = []

        for item in items_list:
            if item.get("is_correct", False) == True:
                correct_items.append(item)
            else:
                incorrect_items.append(item)

        # 确保输出目录存在
        Path(output_correct).parent.mkdir(parents=True, exist_ok=True)
        Path(output_incorrect).parent.mkdir(parents=True, exist_ok=True)

        # 保存正确项（不包含外层方括号）
        with open(output_correct, 'w', encoding='utf-8') as f:
            json_content = json.dumps(correct_items, indent=2, ensure_ascii=False)
            # 移除首尾的方括号
            if json_content.startswith('[') and json_content.endswith(']'):
                json_content = json_content[1:-1]
            f.write(json_content)

        # 保存错误项（不包含外层方括号）
        with open(output_incorrect, 'w', encoding='utf-8') as f:
            json_content = json.dumps(incorrect_items, indent=2, ensure_ascii=False)
            # 移除首尾的方括号
            if json_content.startswith('[') and json_content.endswith(']'):
                json_content = json_content[1:-1]
            f.write(json_content)

        # 打印处理结果
        print(f"\n处理完成:")
        print(f"  ✓ 总项目数: {len(items_list)}")
        print(f"  ✓ 正确项: {len(correct_items)} ({len(correct_items) / len(items_list) * 100:.1f}%)")
        print(f"  ✓ 错误项: {len(incorrect_items)} ({len(incorrect_items) / len(items_list) * 100:.1f}%)")
        print(f"\n输出文件:")
        print(f"  - 正确项保存到: {output_correct}")
        print(f"  - 错误项保存到: {output_incorrect}")

        return True

    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"错误: 解析JSON文件失败 - {e}")
        return False
    except Exception as e:
        print(f"错误: 处理文件 {input_file} 时出错 - {e}")
        return False


def generate_output_paths(input_file):
    """
    根据输入文件路径自动生成输出文件路径

    Args:
        input_file: 输入文件路径

    Returns:
        (correct_path, incorrect_path): 正确和错误项的输出路径
    """
    input_path = Path(input_file)
    parent_dir = input_path.parent
    stem = input_path.stem
    suffix = input_path.suffix

    correct_path = parent_dir / f"{stem}_correct{suffix}"
    incorrect_path = parent_dir / f"{stem}_incorrect{suffix}"

    return str(correct_path), str(incorrect_path)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='根据is_correct字段将JSON文件拆分为正确和错误两个文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 指定所有文件路径
  %(prog)s --input input.json --correct correct.json --incorrect incorrect.json

  # 只指定输入文件（自动生成输出文件名）
  %(prog)s --input optimal_solutions.json

  # 使用短参数
  %(prog)s -i input.json -c correct.json -e incorrect.json

自动生成输出文件名规则:
  输入: optimal_solutions.json
  正确项: optimal_solutions_correct.json
  错误项: optimal_solutions_incorrect.json
        """
    )

    # 必需参数
    parser.add_argument('-i', '--input',
                        required=True,
                        help='输入JSON文件路径')

    # 可选参数
    parser.add_argument('-c', '--output_correct',
                        help='正确项输出文件路径（可选，默认自动生成）')
    parser.add_argument('-e', '--output_incorrect',
                        help='错误项输出文件路径（可选，默认自动生成）')

    # 其他选项
    parser.add_argument('-v', '--version',
                        action='version',
                        version='%(prog)s 1.0')
    parser.add_argument('--keep-brackets',
                        action='store_true',
                        help='保留输出JSON的外层方括号')

    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()

    # 打印开始信息
    print("=" * 50)
    print("JSON文件拆分工具")
    print("=" * 50)

    # 确定输出文件路径
    if args.output_correct and args.output_incorrect:
        # 使用用户指定的路径
        output_correct = args.output_correct
        output_incorrect = args.output_incorrect
    else:
        # 自动生成输出路径
        output_correct, output_incorrect = generate_output_paths(args.input)

        # 如果用户只指定了其中一个，使用指定的
        if args.correct:
            output_correct = args.output_correct
        if args.incorrect:
            output_incorrect = args.output_incorrect

        print(f"自动生成输出文件名:")
        if not args.correct:
            print(f"  - 正确项: {output_correct}")
        if not args.incorrect:
            print(f"  - 错误项: {output_incorrect}")
        print()

    # 执行拆分
    success = split_by_correctness(
        args.input,
        output_correct,
        output_incorrect
    )

    if success:
        print("\n✅ 所有文件处理成功！")
        sys.exit(0)
    else:
        print("\n❌ 处理失败！")
        sys.exit(1)


if __name__ == "__main__":
    main()