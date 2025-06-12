#coding:gbk
import json
import argparse
import os
import logging
from typing import Dict, List, Any, Optional
from collections import OrderedDict
import random

# ������־
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_text(previous_text: str, text: str) -> str:
    """�ϲ�previous_text��text���ݣ����previous_text��������ӿո�ָ�"""
    if previous_text and previous_text.strip():
        return f"{previous_text.strip()} {text.strip()}"
    return text.strip()


def find_generation_by_length(generations: List[Dict[str, Any]],
                              maximum: bool = True) -> Optional[Dict[str, Any]]:
    """
    ����total_length������С��generation

    Args:
        generations: ���ɽ���б�
        maximum: ���ΪTrue��������ģ����ΪFalse��������̵�

    Returns:
        ���ҵ���generation������б�Ϊ���򷵻�None
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
    �������ݲ������µ����ݼ�

    Args:
        optimal_incorrect_path: optimal_solutions_4k_incorrect�ļ�·��
        error_pools_path: error_thinking_pools�ļ�·��
        output_path: ����ļ�·��
        debug: �Ƿ����õ�����־
    """
    # ������־����
    if debug:
        logger.setLevel(logging.DEBUG)

    # ����ļ��Ƿ����
    for file_path in [optimal_incorrect_path, error_pools_path]:
        if not os.path.exists(file_path):
            logger.error(f"�ļ�������: {file_path}")
            raise FileNotFoundError(f"�ļ�������: {file_path}")

    try:
        # ���������ļ�
        logger.info("���ڼ��������ļ�...")

        with open(optimal_incorrect_path, "r", encoding="utf-8") as f:
            optimal_incorrect_data = json.load(f)

        with open(error_pools_path, "r", encoding="utf-8") as f:
            error_pools = json.load(f)

        logger.info("�����ļ��������")

        result_data = []

        # ת��error_poolsΪ�ֵ���ʽ���������
        logger.info("������������...")
        error_dict = {item["problem"]: item for item in error_pools}

        # ����ÿ������
        logger.info("��ʼ��������...")
        total_problems = len(optimal_incorrect_data)
        processed_count = 0
        skipped_count = 0

        for i, problem_data in enumerate(optimal_incorrect_data, 1):
            problem = problem_data["problem"]
            solution = problem_data["solution"]  # ��ȡ�����solution

            logger.debug(f"�������� {i}/{total_problems}: {problem[:50]}...")

            chosen_content = ""
            rejected_content = ""

            if problem in error_dict and "model_generations" in error_dict[problem]:
                error_generations = error_dict[problem].get("model_generations", [])

                if error_generations:
                    # ��solution��Ϊһ����ѡ���������뵽���ɽ����
                    solution_generation = {
                        "previous_text": "",
                        "text": solution,
                        "total_length": len(solution)
                    }
                    all_generations = error_generations + [solution_generation]

                    # �ҵ���̵����ɽ����Ϊchosen
                    chosen_generation = find_generation_by_length(all_generations, maximum=False)
                    if chosen_generation:
                        chosen_content = merge_text(
                            chosen_generation.get("previous_text", ""),
                            chosen_generation.get("text", "")
                        )

                    # ��ʣ������ɽ�������ѡ��һ����Ϊrejected
                    remaining_generations = [gen for gen in all_generations if gen != chosen_generation]
                    if remaining_generations:
                        rejected_generation = random.choice(remaining_generations)
                        rejected_content = merge_text(
                            rejected_generation.get("previous_text", ""),
                            rejected_generation.get("text", "")
                        )

            # ֻ�е�chosen��rejected��������ʱ����ӵ����
            if chosen_content and rejected_content:
                result_item = OrderedDict([
                    ("problem", problem),
                    ("chosen", chosen_content),
                    ("rejected", rejected_content)
                ])
                result_data.append(result_item)
                processed_count += 1
                logger.debug(f"���� {i} ������ɣ����ҵ�chosen��rejected����")
            else:
                logger.warning(
                    f"�������� {i}����Ϊδ�ҵ�������chosen��rejected����: chosen={bool(chosen_content)}, rejected={bool(rejected_content)}")
                skipped_count += 1

            # ��ʾ����
            if i % 10 == 0 or i == total_problems:
                logger.info(f"�Ѵ��� {i}/{total_problems} ������ ({i / total_problems * 100:.1f}%)")

        # �����������ļ�
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        logger.info(
            f"������ɣ������� {total_problems} �����⣬���� {processed_count} ����Ч���⣬���� {skipped_count} ����Ч���⡣����ѱ��浽 {output_path}")

        return result_data

    except Exception as e:
        logger.error(f"��������ʱ��������: {str(e)}", exc_info=True)
        raise


def main():
    """�����������������в������������ݴ�����"""
    parser = argparse.ArgumentParser(description="�������ݼ��������µ��ļ�")

    parser.add_argument("--optimal", type=str, default="",
                        help="optimal_solutions_4k_correct/incorrect�ļ�·��")
    parser.add_argument("--pooling", type=str, default="",
                        help="thinking_pools�ļ�·��")
    parser.add_argument("--output", type=str, default="",
                        help="����ļ�·��")
    parser.add_argument("--debug", action="store_true",
                        help="�Ƿ����õ�����־")

    args = parser.parse_args()

    # ��������
    process_data(
        optimal_incorrect_path=args.optimal_incorrect,
        error_pools_path=args.error,
        output_path=args.output,
        debug=args.debug
    )


if __name__ == "__main__":
    main()