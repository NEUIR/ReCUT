#coding:gbk
import json
import argparse
import os
import logging
from typing import Dict, List, Any, Optional
from collections import OrderedDict
import random

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_text(previous_text: str, text: str) -> str:
    """合并previous_text和text内容，如果previous_text存在则添加空格分隔"""
    if previous_text and previous_text.strip():
        return f"{previous_text.strip()} {text.strip()}"
    return text.strip()


def find_generation_by_length(generations: List[Dict[str, Any]],
                              maximum: bool = True) -> Optional[Dict[str, Any]]:
    """
    查找total_length最大或最小的generation

    Args:
        generations: 生成结果列表
        maximum: 如果为True，查找最长的；如果为False，查找最短的

    Returns:
        查找到的generation，如果列表为空则返回None
    """
    if not generations:
        return None

    if maximum:
        return max(generations, key=lambda x: x.get("total_length", 0))
    else:
        return min(generations, key=lambda x: x.get("total_length", float("inf")))


def process_data(optimal_incorrect_path: str = "optimal_solutions_4k_incorrect.json",
                 error_pools_path: str = "error_thinking_pools_5.json",
                 output_path: str = "processed_dataset.json",
                 debug: bool = False) -> None:
    """
    处理数据并生成新的数据集

    Args:
        optimal_incorrect_path: optimal_solutions_4k_incorrect文件路径
        error_pools_path: error_thinking_pools文件路径
        output_path: 输出文件路径
        debug: 是否启用调试日志
    """
    # 设置日志级别
    if debug:
        logger.setLevel(logging.DEBUG)

    # 检查文件是否存在
    for file_path in [optimal_incorrect_path, error_pools_path]:
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            raise FileNotFoundError(f"文件不存在: {file_path}")

    try:
        # 加载数据文件
        logger.info("正在加载数据文件...")

        with open(optimal_incorrect_path, "r", encoding="utf-8") as f:
            optimal_incorrect_data = json.load(f)

        with open(error_pools_path, "r", encoding="utf-8") as f:
            error_pools = json.load(f)

        logger.info("数据文件加载完成")

        result_data = []

        # 转换error_pools为字典形式，方便查找
        logger.info("构建问题索引...")
        error_dict = {item["problem"]: item for item in error_pools}

        # 处理每个问题
        logger.info("开始处理问题...")
        total_problems = len(optimal_incorrect_data)
        processed_count = 0
        skipped_count = 0

        for i, problem_data in enumerate(optimal_incorrect_data, 1):
            problem = problem_data["problem"]
            solution = problem_data["solution"]  # 获取问题的solution

            logger.debug(f"处理问题 {i}/{total_problems}: {problem[:50]}...")

            chosen_content = ""
            rejected_content = ""

            if problem in error_dict and "model_generations" in error_dict[problem]:
                error_generations = error_dict[problem].get("model_generations", [])

                if error_generations:
                    # 将solution作为一个候选方案，加入到生成结果中
                    solution_generation = {
                        "previous_text": "",
                        "text": solution,
                        "total_length": len(solution)
                    }
                    all_generations = error_generations + [solution_generation]

                    # 找到最短的生成结果作为chosen
                    chosen_generation = find_generation_by_length(all_generations, maximum=False)
                    if chosen_generation:
                        chosen_content = merge_text(
                            chosen_generation.get("previous_text", ""),
                            chosen_generation.get("text", "")
                        )

                    # 从剩余的生成结果中随机选择一个作为rejected
                    remaining_generations = [gen for gen in all_generations if gen != chosen_generation]
                    if remaining_generations:
                        rejected_generation = random.choice(remaining_generations)
                        rejected_content = merge_text(
                            rejected_generation.get("previous_text", ""),
                            rejected_generation.get("text", "")
                        )

            # 只有当chosen和rejected都有内容时才添加到结果
            if chosen_content and rejected_content:
                result_item = OrderedDict([
                    ("problem", problem),
                    ("chosen", chosen_content),
                    ("rejected", rejected_content)
                ])
                result_data.append(result_item)
                processed_count += 1
                logger.debug(f"问题 {i} 处理完成，已找到chosen和rejected内容")
            else:
                logger.warning(
                    f"跳过问题 {i}，因为未找到完整的chosen和rejected内容: chosen={bool(chosen_content)}, rejected={bool(rejected_content)}")
                skipped_count += 1

            # 显示进度
            if i % 10 == 0 or i == total_problems:
                logger.info(f"已处理 {i}/{total_problems} 个问题 ({i / total_problems * 100:.1f}%)")

        # 保存结果到新文件
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"处理完成！共处理 {total_problems} 个问题，保存 {processed_count} 个有效问题，跳过 {skipped_count} 个无效问题。结果已保存到 {output_path}")

        return result_data

    except Exception as e:
        logger.error(f"处理数据时发生错误: {str(e)}", exc_info=True)
        raise


def main():
    """主函数，处理命令行参数并调用数据处理函数"""
    parser = argparse.ArgumentParser(description="处理数据集并生成新的文件")

    parser.add_argument("--optimal", type=str, default="",
                        help="optimal_solutions_4k_correct/incorrect文件路径")
    parser.add_argument("--pooling", type=str, default="",
                        help="thinking_pools文件路径")
    parser.add_argument("--output", type=str, default="",
                        help="输出文件路径")
    parser.add_argument("--debug", action="store_true",
                        help="是否启用调试日志")

    args = parser.parse_args()

    # 处理数据
    process_data(
        optimal_incorrect_path=args.optimal_incorrect,
        error_pools_path=args.error,
        output_path=args.output,
        debug=args.debug
    )


if __name__ == "__main__":
    main()