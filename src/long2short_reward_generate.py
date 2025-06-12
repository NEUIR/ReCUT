#coding:utf-8
import os
import re
import torch
import json
import time
import gc
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
from click import prompt
from vllm import LLM, SamplingParams
import numpy as np
from transformers import AutoTokenizer

import random
# 设置环境变量
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_model(model_path):
    """
    初始化并返回LLM模型实例，使用优化的配置
    :param model_path: 模型路径
    :return: 初始化好的LLM实例
    """

    return LLM(
        model=model_path,
        dtype="float16",
        max_model_len=8192,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=4,
        trust_remote_code=True,
        # 以下参数可能需要根据vLLM版本与具体模型调整
        enforce_eager=False,  # 如可用，使用CUDA图优化
        # KV缓存优化
        max_num_batched_tokens=8192,  # 增加批处理token数量
    )


def extract_final_answer(generated_text):
    """从生成的文本中提取最终答案"""
    if not generated_text:
        return None

    # 方法1: 在"Final answer"之后查找 \boxed{...}
    final_section_pattern = r'(?:Final answer|FINAL ANSWER|final answer).*?\\boxed\{(.*)\}'
    final_section_matches = re.findall(final_section_pattern, generated_text)

    if final_section_matches:
        return final_section_matches[-1].strip()  # 返回最后一个匹配

    # 方法2: 查找最后一个 \boxed{...}
    boxed_pattern = r'\\boxed\{(.*)\}'
    boxed_matches = re.findall(boxed_pattern, generated_text)

    if boxed_matches:
        return boxed_matches[-1].strip()

    return None


def batch_inference(model, prompts: List[str], sampling_params: SamplingParams) -> List[Dict]:
    """
    执行批量推理以提高吞吐量
    :param model: 初始化好的LLM实例
    :param prompts: 要处理的提示列表
    :param sampling_params: 采样参数
    :return: 推理结果列表
    """
    if not prompts:
        return []

    try:
        start_time = time.time()
        outputs = model.generate(prompts, sampling_params)
        elapsed_time = time.time() - start_time

        results = []
        for i, output in enumerate(outputs):
            if output and len(output.outputs) > 0:
                results.append({
                    "prompt": prompts[i],
                    "generated_text": output.outputs[0].text,
                    "execution_time": elapsed_time / len(prompts)
                })
            else:
                results.append({
                    "prompt": prompts[i],
                    "generated_text": None,
                    "error": "未生成输出",
                    "execution_time": elapsed_time / len(prompts)
                })

        return results
    except Exception as e:
        print(f"批量推理发生错误: {str(e)}")
        return [{
            "prompt": prompt,
            "generated_text": None,
            "error": str(e),
            "execution_time": time.time() - start_time
        } for prompt in prompts]


def optimized_inference_with_step_thinking(
        model,
        problem: Union[str, List[str]],
        previous_steps: Optional[Union[List[str], List[List[str]]]] = None,
        mode: Optional[Union[str, List[str]]] = None,
        batch_size: int = 4,
        tokenizer=None  # 添加tokenizer参数
) -> Union[Dict, List[Dict]]:
    """
    优化的推理函数，支持批处理多个问题
    :param model: 初始化好的LLM实例
    :param problem: 问题文本或问题文本列表
    :param previous_steps: 之前的思考步骤或步骤列表的列表
    :param mode: 思考模式或模式列表
    :param batch_size: 批处理大小
    :param tokenizer: 用于格式化提示的分词器
    :return: 包含生成内容的字典或字典列表
    """
    # 将单个输入转换为列表以进行批处理
    is_single_input = not isinstance(problem, list)

    problems = [problem] if is_single_input else problem

    if previous_steps is None:
        previous_steps_list = [[] for _ in range(len(problems))]
    elif is_single_input:
        previous_steps_list = [previous_steps]
    else:
        previous_steps_list = previous_steps

    if mode is None:
        modes = ["short" for _ in range(len(problems))]
    elif is_single_input:
        modes = [mode]
    else:
        modes = mode

    # 准备提示
    prompts = []
    for prob, prev_steps, md in zip(problems, previous_steps_list, modes):
        current_step = len(prev_steps) + 1
        previous_text = ' '.join(prev_steps)

        # 优化的提示 - 更短更直接
        if md == "long":
            user_prompt = (
                f"Please think step by step to solve this problem based on the previous reasoning, and think deeply about each step.\n"
                f"Problem:\n{prob}\n\n"
                f"Previous reasoning:\n{previous_text}\n\n"
                f"Your answer must follow the format below:\n"
                f"[STEP{current_step}]: ...\n"
                f"[STEP{current_step + 1}]: ...\n"
                f"...\n"
                f"[STEPn]: Final reasoning...(Note: n represents the number of your last step of reasoning.)\n" 
                f"Final answer: \\boxed{{answer}}\n\n"
                f"Now continue thinking from [STEP{current_step}]."
            )
        else:
            user_prompt = (
                f"Please think step by step to solve this problem based on the previous reasoning, and each step should be concise but critical.\n"
                f"Problem:\n{prob}\n\n"
                f"Previous reasoning:\n{previous_text}\n\n"
                f"Your answer must follow the format below:\n"
                f"[STEP{current_step}]: ...\n"
                f"[STEP{current_step + 1}]: ...\n"
                f"...\n"
                f"[STEPn]: Final reasoning...(Note: n represents the number of your last step of reasoning.)\n"    
                f"Final answer: \\boxed{{answer}}\n\n"
                f"Now continue thinking from [STEP{current_step}]."
            )
        # 将提示包装成聊天格式
        chat_prompt = [{"role": "user", "content": user_prompt}]
        # 应用聊天模板
        formatted_prompt = tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
        prompts.append(formatted_prompt)

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=2048
    )

    # 分批处理 - 不再限制批次大小，直接使用用户指定的批次大小
    all_results = []

    # 对批次运行推理
    batch_outputs = batch_inference(model, prompts, sampling_params)

    # 处理结果
    for j, output in enumerate(batch_outputs):
        if j >= len(problems):
            continue

        p = problems[j]
        ps = previous_steps_list[j]
        m = modes[j]
        current_step = len(ps) + 1

        generated_text = output.get("generated_text", "")

        # 提取步骤内容
        if generated_text:
            # 更新后的正则表达式，捕获步骤标识和内容两个组
            pattern = rf'(?i)((?:[\[\(\{{\<\*\#\-\_\.]+)?\s*(?:step\s*[-_.\s]*{current_step}|{current_step}[\.\):])(?:[\]\)\}}\>\*\#\-\_\.]+)?:?\s*|\n\n{current_step}\.\s*|\n{current_step}\.\s*)(.*?)(?=(?:[\[\(\{{\<\*\#\-\_\.]+)?\s*(?:step\s*[-_.\s]*{current_step + 1}|{current_step + 1}[\.\):])(?:[\]\)\}}\>\*\#\-\_\.]+)?:?|\n\n{current_step + 1}\.|\n{current_step + 1}\.|\s*(?:[\[\(\{{\<\*\#\-\_\.]+)?\s*final\s*answer\s*(?:[\]\)\}}\>\*\#\-\_\.]+)?:?|\Z)'
            matches = re.search(pattern, generated_text, re.DOTALL)

            if matches:
                step_identifier = matches.group(1).strip()  # 提取步骤标识
                step_content = matches.group(2).strip()  # 提取步骤内容
                # 组合完整步骤
                new_step_content = f"{step_identifier} {step_content}"
            else:
                new_step_content = None

            final_answer = extract_final_answer(generated_text)

            result = {
                "prompt": prompts[j],
                "generated_text": generated_text,
                "next_step": new_step_content,
                "final_answer": final_answer,
                "execution_time": output.get("execution_time", 0)
            }
        else:
            result = {
                "prompt": prompts[j],
                "generated_text": None,
                "next_step": None,
                "final_answer": None,
                "error": output.get("error", "未生成输出"),
                "execution_time": output.get("execution_time", 0)
            }

        all_results.append(result)

    # 如果输入是单个的，则返回单个结果
    return all_results[0] if is_single_input else all_results

def calculate_reward(generated_text, expected_answer, alpha=1.0, beta=1.0):
    """
    计算奖励分数
    :param generated_text: 模型生成的文本
    :param expected_answer: 期望的答案
    :param alpha: 正确答案的奖励系数
    :param beta: 错误答案的惩罚系数
    :return: 奖励分数和指示函数值
    """
    # 提取最终答案
    final_answer = extract_final_answer(generated_text)

    # 计算文本长度 L(n_y)
    text_length = len(generated_text)

    # 防止除零错误
    if text_length == 0:
        return -float('inf'), 0

    # 计算指示函数 I(n_y)
    indicator = 1 if final_answer and final_answer.strip() == expected_answer.strip() else 0

    # 应用奖励公式 R(n_y) = [α?I(n_y) - β?(1-I(n_y))] / L(n_y)
    reward = (alpha * indicator - beta * (1 - indicator)) / text_length

    return reward, indicator


def generate_dataset_batched(model, questions, num_iterations=5, alpha=1.0, beta=1.0, batch_size=4, tokenizer=None):
    """
    带批处理优化的数据集生成，并增加错误思维池
    :param model: 初始化好的LLM实例
    :param questions: 问题列表，每个问题是一个包含'problem'和'answer'字段的字典
    :param num_iterations: 最大迭代次数
    :param alpha: 正确答案的奖励系数
    :param beta: 错误答案的惩罚系数
    :param batch_size: 每批处理的问题数量
    :param tokenizer: 用于格式化提示的分词器
    :return: 正确思维池结果、错误思维池结果和选择的步骤结果
    """
    thinking_pools = []  # 正确思维池
    error_thinking_pools = []  # 新增：错误思维池
    selected_results = []

    # 计算总批次数
    total_batches = (len(questions) + batch_size - 1) // batch_size
    print(f"总共有 {len(questions)} 个问题，将分成 {total_batches} 个批次处理，每批次 {batch_size} 个问题")

    # 按批次处理问题以提高效率
    for batch_index in range(total_batches):
        batch_start = batch_index * batch_size
        batch_end = min(batch_start + batch_size, len(questions))
        batch_questions = questions[batch_start:batch_end]

        print(f"开始处理第 {batch_index + 1}/{total_batches} 批次，包含 {len(batch_questions)} 个问题...")

        # 准备问题和预期答案
        batch_problems = []
        batch_expected_answers = []

        for question_data in batch_questions:
            problem = question_data.get('problem')
            expected_answer = question_data.get('answer')

            if not problem or not expected_answer:
                print(f"跳过缺失问题或答案的数据项")
                continue

            batch_problems.append(problem)
            batch_expected_answers.append(expected_answer)

        # 验证批次数据
        if not batch_problems:
            print(f"批次 {batch_index + 1} 没有有效问题，跳过")
            continue

        # 为批次中的每个问题初始化数据结构
        batch_previous_steps = [[] for _ in range(len(batch_problems))]
        batch_thinking_pools = [{
            'problem': problem,
            'expected_answer': expected_answer,
            'model_generations': []
        } for problem, expected_answer in zip(batch_problems, batch_expected_answers)]

        # 新增：错误思维池初始化
        batch_error_thinking_pools = [{
            'problem': problem,
            'expected_answer': expected_answer,
            'model_generations': []
        } for problem, expected_answer in zip(batch_problems, batch_expected_answers)]

        batch_selected_results = [{
            'problem': problem,
            'expected_answer': expected_answer,
            'selected_steps': [],
            'thinking_modes': [],
            'total_length': 0,
            'final_answer': None
        } for problem, expected_answer in zip(batch_problems, batch_expected_answers)]

        # 为每个问题跟踪思考模式历史
        batch_thinking_modes_history = [[] for _ in range(len(batch_problems))]
        # 为每个问题跟踪selected_steps历史
        batch_selected_steps_history = [[] for _ in range(len(batch_problems))]

        batch_found_answers = [False for _ in range(len(batch_problems))]

        # 遍历步骤
        for step_num in range(1, num_iterations + 1):
            print(f"生成批次 {batch_index + 1}/{total_batches} 的步骤 {step_num}...")

            # 跳过已有答案的问题
            active_indices = [i for i, found in enumerate(batch_found_answers) if not found]
            if not active_indices:
                print(f"批次 {batch_index + 1} 中的所有问题均已解决。移至下一批次。")
                break

            # 为长思考模式和短思考模式准备问题
            long_problems = []
            long_previous_steps = []
            short_problems = []
            short_previous_steps = []
            active_to_orig_indices = {}  # 映射处理索引到原始索引

            # 为每个活跃问题添加长模式和短模式的两个版本
            for idx, i in enumerate(active_indices):
                problem = batch_problems[i]
                previous_steps = batch_previous_steps[i]

                # 为长模式添加
                long_problems.append(problem)
                long_previous_steps.append(previous_steps)
                active_to_orig_indices[len(long_problems) - 1] = i

                # 为短模式添加
                short_problems.append(problem)
                short_previous_steps.append(previous_steps)

            # 长思考模式批量推理
            long_modes = ["long"] * len(long_problems)
            long_results = optimized_inference_with_step_thinking(
                model=model,
                problem=long_problems,
                previous_steps=long_previous_steps,
                mode=long_modes,
                batch_size=batch_size,
                tokenizer=tokenizer  # 传入tokenizer
            )

            # 确保结果是列表
            if not isinstance(long_results, list):
                long_results = [long_results]

            # 短思考模式批量推理
            short_modes = ["short"] * len(short_problems)
            short_results = optimized_inference_with_step_thinking(
                model=model,
                problem=short_problems,
                previous_steps=short_previous_steps,
                mode=short_modes,
                batch_size=batch_size,
                tokenizer=tokenizer  # 传入tokenizer
            )

            # 确保结果是列表
            if not isinstance(short_results, list):
                short_results = [short_results]

            # 处理每个活跃问题的结果
            for idx, active_idx in enumerate(active_indices):
                if idx < len(long_results) and idx < len(short_results):
                    problem = batch_problems[active_idx]
                    expected_answer = batch_expected_answers[active_idx]
                    previous_steps = batch_previous_steps[active_idx]
                    thinking_pool = batch_thinking_pools[active_idx]
                    # 新增：错误思维池
                    error_thinking_pool = batch_error_thinking_pools[active_idx]
                    selected_result = batch_selected_results[active_idx]
                    # 获取当前的思考模式历史
                    current_thinking_modes_history = batch_thinking_modes_history[active_idx]
                    # 获取当前的selected_steps历史
                    current_selected_steps = batch_selected_steps_history[active_idx]

                    # 获取长思考和短思考结果
                    long_result = long_results[idx]
                    short_result = short_results[idx]

                    # 确保previous_text不是None
                    previous_text = ' '.join(previous_steps) if previous_steps else ""

                    # 提取最终答案
                    long_final_answer = long_result.get("final_answer")
                    short_final_answer = short_result.get("final_answer")

                    # 计算总长度 - 添加对None值的检查
                    long_generated_text = long_result.get("generated_text", "")
                    long_generated_text = "" if long_generated_text is None else long_generated_text
                    long_total_length = len(previous_text) + len(long_generated_text)

                    short_generated_text = short_result.get("generated_text", "")
                    short_generated_text = "" if short_generated_text is None else short_generated_text
                    short_total_length = len(previous_text) + len(short_generated_text)

                    # 计算奖励
                    long_reward, long_correct = calculate_reward(
                        long_generated_text,
                        expected_answer,
                        alpha,
                        beta
                    )

                    short_reward, short_correct = calculate_reward(
                        short_generated_text,
                        expected_answer,
                        alpha,
                        beta
                    )

                    # 处理长思考模式的结果
                    if long_final_answer:
                        # 为当前步骤创建完整的思考模式历史
                        complete_thinking_modes = current_thinking_modes_history.copy()
                        complete_thinking_modes.append('long')

                        # 复制当前的selected_steps历史
                        complete_selected_steps = current_selected_steps.copy()

                        # 为当前步骤添加完整的生成文本
                        if long_generated_text:
                            complete_selected_steps.append(long_generated_text)

                        # 判断是否为正确答案
                        if long_final_answer.strip() == expected_answer.strip():
                            # 添加到正确思维池
                            thinking_pool['model_generations'].append({
                                'step_num': step_num,
                                'mode': 'long',
                                'text': long_generated_text,
                                'previous_text': previous_text,
                                'total_length': long_total_length,
                                'final_answer': long_final_answer,
                                'thinking_modes': complete_thinking_modes,
                                'selected_steps': complete_selected_steps
                            })
                        else:
                            # 新增：添加到错误思维池
                            error_thinking_pool['model_generations'].append({
                                'step_num': step_num,
                                'mode': 'long',
                                'text': long_generated_text,
                                'previous_text': previous_text,
                                'total_length': long_total_length,
                                'final_answer': long_final_answer,
                                'thinking_modes': complete_thinking_modes,
                                'selected_steps': complete_selected_steps
                            })

                    # 处理短思考模式的结果
                    if short_final_answer:
                        # 为当前步骤创建完整的思考模式历史
                        complete_thinking_modes = current_thinking_modes_history.copy()
                        complete_thinking_modes.append('short')

                        # 复制当前的selected_steps历史
                        complete_selected_steps = current_selected_steps.copy()

                        # 为当前步骤添加完整的生成文本
                        if short_generated_text:
                            complete_selected_steps.append(short_generated_text)

                        # 判断是否为正确答案
                        if short_final_answer.strip() == expected_answer.strip():
                            # 添加到正确思维池
                            thinking_pool['model_generations'].append({
                                'step_num': step_num,
                                'mode': 'short',
                                'text': short_generated_text,
                                'previous_text': previous_text,
                                'total_length': short_total_length,
                                'final_answer': short_final_answer,
                                'thinking_modes': complete_thinking_modes,
                                'selected_steps': complete_selected_steps
                            })
                        else:
                            # 新增：添加到错误思维池
                            error_thinking_pool['model_generations'].append({
                                'step_num': step_num,
                                'mode': 'short',
                                'text': short_generated_text,
                                'previous_text': previous_text,
                                'total_length': short_total_length,
                                'final_answer': short_final_answer,
                                'thinking_modes': complete_thinking_modes,
                                'selected_steps': complete_selected_steps
                            })

                    # 根据奖励选择步骤
                    chosen_result = long_result if long_reward > short_reward else short_result
                    chosen_mode = 'long' if long_reward > short_reward else 'short'

                    # 更新思考模式历史
                    batch_thinking_modes_history[active_idx].append(chosen_mode)

                    # 更新之前的步骤
                    next_step_content = chosen_result.get("next_step")
                    if next_step_content:
                        batch_previous_steps[active_idx].append(next_step_content)

                        # 更新选择的结果
                        selected_result['selected_steps'].append(next_step_content)
                        selected_result['thinking_modes'].append(chosen_mode)
                        selected_result['total_length'] += len(next_step_content)

                        # 更新selected_steps历史
                        batch_selected_steps_history[active_idx].append(next_step_content)

                        # 检查最终答案
                        step_final_answer = extract_final_answer(next_step_content)
                        if step_final_answer:
                            batch_found_answers[active_idx] = True
                            selected_result['final_answer'] = step_final_answer
                            print(
                                f"在批次 {batch_index + 1} 的步骤 {step_num} 中找到问题 {batch_start + active_idx + 1} 的最终答案: {step_final_answer}")

        # 最终处理批次结果
        for idx, (thinking_pool, error_thinking_pool, selected_result, previous_steps) in enumerate(
                zip(batch_thinking_pools, batch_error_thinking_pools, batch_selected_results, batch_previous_steps)):
            selected_result['previous_text'] = ' '.join(previous_steps)
            thinking_pools.append(thinking_pool)
            error_thinking_pools.append(error_thinking_pool)  # 添加错误思维池到结果
            selected_results.append(selected_result)

        # 释放内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"完成批次 {batch_index + 1}/{total_batches} 的处理")

    return thinking_pools, error_thinking_pools, selected_results


def select_optimal_solutions(thinking_pools, selected_results):
    """
    从思维池和选择步骤结果中选择每个问题的最优解（正确且最短）
    并保存正确率到最终结果中

    :param thinking_pools: 思维池结果列表
    :param selected_results: 选择的步骤结果列表
    :return: 最优解列表和元数据
    """
    optimal_solutions = []

    # 确保思维池和选择步骤结果列表长度相同
    assert len(thinking_pools) == len(selected_results), "思维池和选择步骤结果列表长度不一致"

    # 创建进度条
    total = len(thinking_pools)

    # 按批次处理，以提高性能
    batch_size = 100
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        print(
            f"处理批次 {batch_start // batch_size + 1}/{(total + batch_size - 1) // batch_size}，问题 {batch_start + 1} 到 {batch_end}...")

        batch_solutions = []
        for i in range(batch_start, batch_end):
            thinking_pool = thinking_pools[i]
            selected_result = selected_results[i]

            problem = thinking_pool['problem']
            # 确保使用原始数据中的答案作为正确答案标准
            expected_answer = thinking_pool['expected_answer']

            print(f"处理问题 {i + 1}/{total}: {problem[:50]}...")
            print(f"预期答案: {expected_answer}")

            # 收集所有正确的解答
            correct_solutions = []

            # 检查选择步骤结果是否正确
            if selected_result.get('final_answer') and selected_result[
                'final_answer'].strip() == expected_answer.strip():
                print(f"选择步骤结果答案正确: {selected_result['final_answer']}")
                correct_solutions.append({
                    'source': 'selected_steps',
                    'content': selected_result['previous_text'],
                    'length': selected_result['total_length'],
                    'thinking_modes': selected_result['thinking_modes'],
                    'selected_steps': selected_result['selected_steps'],
                    'final_answer': selected_result['final_answer']
                })

            # 检查思维池中的所有生成结果
            for generation in thinking_pool.get('model_generations', []):
                if generation.get('final_answer') and generation['final_answer'].strip() == expected_answer.strip():
                    print(
                        f"思维池结果答案正确: {generation['final_answer']} (步骤 {generation['step_num']}, 模式 {generation['mode']})")
                    combined_text = (
                            generation.get('previous_text', '') + ' ' + generation['text']).strip() if generation.get(
                        'previous_text') else generation['text']
                    correct_solutions.append({
                        'source': f"thinking_pool_step{generation['step_num']}_{generation['mode']}",
                        'content': combined_text,
                        'length': generation['total_length'],
                        'final_answer': generation['final_answer'],
                        'thinking_modes': generation.get('thinking_modes', []),  # 包含思考模式历史
                        'selected_steps': generation.get('selected_steps', [])  # 包含选择步骤历史
                    })

            # 如果有正确解答，选择最短的
            if correct_solutions:
                shortest_solution = min(correct_solutions, key=lambda x: x['length'])
                print(f"找到最优解: 来源={shortest_solution['source']}, 长度={shortest_solution['length']}")

                # 创建最优解结果
                optimal_solution = {
                    'problem': problem,
                    'expected_answer': expected_answer,  # 保存原始预期答案
                    'source': shortest_solution['source'],
                    'solution': shortest_solution['content'],
                    'result_length': shortest_solution['length'],
                    'final_answer': shortest_solution['final_answer'],
                    'thinking_modes': shortest_solution.get('thinking_modes', []),  # 添加思考模式历史
                    'selected_steps': shortest_solution.get('selected_steps', []),  # 添加选择步骤历史
                    'is_correct': True  # 标记为正确答案
                }

                batch_solutions.append(optimal_solution)
            else:
                print(f"警告: 问题 {i + 1} 没有找到正确答案，使用选择步骤结果")
                # 如果没有正确解答，使用选择步骤结果
                optimal_solution = {
                    'problem': problem,
                    'expected_answer': expected_answer,  # 保存原始预期答案
                    'source': 'selected_steps_incorrect',
                    'thinking_modes': selected_result.get('thinking_modes', []),
                    'selected_steps': selected_result.get('selected_steps', []),
                    'solution': selected_result.get('previous_text', ''),
                    'result_length': selected_result.get('total_length', 0),
                    'final_answer': selected_result.get('final_answer'),
                    'is_correct': False  # 标记为错误答案
                }
                batch_solutions.append(optimal_solution)

        # 添加批次处理结果到总结果
        optimal_solutions.extend(batch_solutions)

        # 阶段性保存
        temp_result = {
            'metadata': {
                'processed_problems': len(optimal_solutions),
                'total_problems': total,
                'correct_count': sum(1 for sol in optimal_solutions if sol.get('is_correct', False)),
                'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'solutions': optimal_solutions
        }

        temp_file = f"./result/optimal_solutions_temp_{batch_start}_{batch_end}.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(temp_result, f, ensure_ascii=False, indent=2)
        print(f"已保存临时结果到 {temp_file}")

    # 计算正确率
    correct_count = sum(1 for sol in optimal_solutions if sol.get('is_correct', False))
    correct_rate = correct_count / len(optimal_solutions) * 100 if optimal_solutions else 0

    # 打印统计信息
    print(f"\n统计信息:")
    print(f"总问题数: {len(optimal_solutions)}")
    print(f"有正确答案的问题数: {correct_count}")
    print(f"正确率: {correct_rate:.2f}%")

    # 创建包含元数据的结果
    final_result = {
        'metadata': {
            'total_problems': len(optimal_solutions),
            'correct_count': correct_count,
            'correct_rate': correct_rate,
            'processing_date': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'solutions': optimal_solutions
    }

    return final_result


def analyze_results(optimal_solutions_path):
    """
    分析结果文件并生成详细报告
    :param optimal_solutions_path: 最优解结果文件路径
    :return: None
    """
    print(f"正在分析结果文件 {optimal_solutions_path}...")

    try:
        with open(optimal_solutions_path, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # 提取元数据
        metadata = results.get('metadata', {})
        solutions = results.get('solutions', [])

        # 基本统计
        total_problems = metadata.get('total_problems', len(solutions))
        correct_count = metadata.get('correct_count', 0)
        correct_rate = metadata.get('correct_rate', 0)

        # 高级统计
        length_stats = {
            'min': float('inf'),
            'max': 0,
            'avg': 0,
            'median': 0
        }

        thinking_modes_counts = {'long': 0, 'short': 0}
        correct_sources = {}
        steps_used = []

        # 新增：思考模式序列统计
        thinking_modes_sequences = {}

        # 收集统计数据
        solution_lengths = []
        for sol in solutions:
            # 长度统计
            length = sol.get('result_length', 0)
            solution_lengths.append(length)

            length_stats['min'] = min(length_stats['min'], length) if length > 0 else length_stats['min']
            length_stats['max'] = max(length_stats['max'], length)

            # 思考模式统计
            thinking_modes = sol.get('thinking_modes', [])
            for mode in thinking_modes:
                thinking_modes_counts[mode] = thinking_modes_counts.get(mode, 0) + 1

            # 新增：思考模式序列统计
            if thinking_modes:
                mode_sequence = ','.join(thinking_modes)
                thinking_modes_sequences[mode_sequence] = thinking_modes_sequences.get(mode_sequence, 0) + 1

            # 正确答案来源统计
            if sol.get('is_correct', False):
                source = sol.get('source', 'unknown')
                correct_sources[source] = correct_sources.get(source, 0) + 1

            # 步骤数统计
            steps_used.append(len(sol.get('selected_steps', [])))

        # 计算平均值和中位数
        length_stats['avg'] = sum(solution_lengths) / len(solution_lengths) if solution_lengths else 0
        length_stats['median'] = np.median(solution_lengths) if solution_lengths else 0

        avg_steps = sum(steps_used) / len(steps_used) if steps_used else 0

        # 生成报告
        report = {
            'basic_stats': {
                'total_problems': total_problems,
                'correct_count': correct_count,
                'correct_rate': correct_rate
            },
            'solution_length_stats': length_stats,
            'thinking_modes_distribution': thinking_modes_counts,
            'thinking_modes_sequences': thinking_modes_sequences,  # 新增：思考模式序列统计
            'correct_solution_sources': correct_sources,
            'steps_stats': {
                'min': min(steps_used) if steps_used else 0,
                'max': max(steps_used) if steps_used else 0,
                'avg': avg_steps
            },
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # 保存报告
        report_path = "./result/analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"分析报告已保存到 {report_path}")

        # 打印主要统计结果
        print("\n=== 结果分析报告 ===")
        print(f"总问题数: {total_problems}")
        print(f"正确答案数: {correct_count}")
        print(f"正确率: {correct_rate:.2f}%")
        print(f"解答长度 - 最小: {length_stats['min']}, 最大: {length_stats['max']}, 平均: {length_stats['avg']:.2f}")
        print(
            f"思考模式分布 - 长模式: {thinking_modes_counts.get('long', 0)}, 短模式: {thinking_modes_counts.get('short', 0)}")
        print(f"平均步骤数: {avg_steps:.2f}")

        # 打印思考模式序列统计
        print("\n思考模式序列分布:")
        for sequence, count in sorted(thinking_modes_sequences.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sequence}: {count}")

        print("===================")

    except Exception as e:
        print(f"分析结果时出错: {str(e)}")


def analyze_error_thinking_pools(error_thinking_pools_path):
    """
    分析错误思维池并生成错误模式报告
    :param error_thinking_pools_path: 错误思维池结果文件路径
    :return: None
    """
    print(f"正在分析错误思维池文件 {error_thinking_pools_path}...")

    try:
        with open(error_thinking_pools_path, 'r', encoding='utf-8') as f:
            error_pools = json.load(f)

        # 基本统计
        total_problems = len(error_pools)
        problems_with_errors = sum(1 for pool in error_pools if pool['model_generations'])
        total_error_generations = sum(len(pool['model_generations']) for pool in error_pools)

        # 错误模式统计
        error_modes = {'long': 0, 'short': 0}
        error_steps = {}  # 统计不同步骤的错误数量
        error_lengths = []  # 收集错误生成的长度

        # 错误答案分析
        error_answers = []  # 收集所有错误答案，用于分析常见错误类型

        # 新增：思考模式序列统计
        error_thinking_modes_sequences = {}

        for pool in error_pools:
            problem = pool['problem']
            expected_answer = pool['expected_answer']

            for generation in pool['model_generations']:
                # 统计错误模式
                mode = generation.get('mode', 'unknown')
                error_modes[mode] = error_modes.get(mode, 0) + 1

                # 统计错误步骤
                step = generation.get('step_num', 0)
                error_steps[step] = error_steps.get(step, 0) + 1

                # 收集错误长度
                error_lengths.append(generation.get('total_length', 0))

                # 收集错误答案
                error_answer = generation.get('final_answer', '')
                if error_answer:
                    error_answers.append({
                        'expected': expected_answer,
                        'generated': error_answer,
                        'problem': problem
                    })

                # 统计错误思考模式序列
                thinking_modes = generation.get('thinking_modes', [])
                if thinking_modes:
                    mode_sequence = ','.join(thinking_modes)
                    error_thinking_modes_sequences[mode_sequence] = error_thinking_modes_sequences.get(mode_sequence,
                                                                                                       0) + 1

        # 计算统计值
        avg_error_length = sum(error_lengths) / len(error_lengths) if error_lengths else 0
        median_error_length = np.median(error_lengths) if error_lengths else 0

        # 生成报告
        report = {
            'basic_stats': {
                'total_problems': total_problems,
                'problems_with_errors': problems_with_errors,
                'error_percentage': (problems_with_errors / total_problems * 100) if total_problems > 0 else 0,
                'total_error_generations': total_error_generations
            },
            'error_mode_distribution': error_modes,
            'error_step_distribution': error_steps,
            'error_length_stats': {
                'min': min(error_lengths) if error_lengths else 0,
                'max': max(error_lengths) if error_lengths else 0,
                'avg': avg_error_length,
                'median': median_error_length
            },
            'error_thinking_modes_sequences': error_thinking_modes_sequences,
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        # 添加错误答案样本（最多10个）
        if error_answers:
            sample_size = min(10, len(error_answers))
            report['error_answer_samples'] = random.sample(error_answers, sample_size)

        # 保存报告
        report_path = "./result/error_analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"错误分析报告已保存到 {report_path}")

        # 打印主要统计结果
        print("\n=== 错误思维池分析报告 ===")
        print(f"总问题数: {total_problems}")
        print(f"含有错误的问题数: {problems_with_errors} ({report['basic_stats']['error_percentage']:.2f}%)")
        print(f"总错误生成数: {total_error_generations}")
        print(f"错误模式分布 - 长模式: {error_modes.get('long', 0)}, 短模式: {error_modes.get('short', 0)}")
        print(f"错误生成长度 - 平均: {avg_error_length:.2f}, 中位数: {median_error_length:.2f}")

        # 打印错误思考模式序列分布（前5个）
        print("\n错误思考模式序列分布 (前5个):")
        for sequence, count in sorted(error_thinking_modes_sequences.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {sequence}: {count}")

        print("===================")

    except Exception as e:
        print(f"分析错误思维池时出错: {str(e)}")

if __name__ == "__main__":
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    # 配置
    model_path = "/data3/xiongqiushi/model/Qwen2.5-14B-Instruct/"
    dataset_path = "./data/deepscaler.json"
    thinking_pools_path = "./instruct_reward_data_14B/thinking_pools_4k.json"
    error_thinking_pools_path = "./instruct_reward_data_14B/error_thinking_pools_4k.json"  # 新增：错误思维池保存路径
    selected_results_path = "./instruct_reward_data_14B/selected_results_4k.json"
    optimal_solutions_path = "./instruct_reward_data_14B/optimal_solutions_4k.json"

    # 创建结果目录
    # os.makedirs("./result", exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        questions = json.load(f)

    # 限制处理的问题数量（可选，注释掉以处理所有问题）
    # questions = questions[:100]  # 只处理前100个问题
    questions = random.sample(questions, 8000)
    print(len(questions))
    print(questions[0])
    # 获取问题总量
    total_questions = len(questions)

    model = setup_model(model_path)
    # 设置批处理大小 - 可以根据GPU内存和问题复杂度调整
    batch_size = 4000  # 可以根据内存情况调整

    # 计算预计批次数
    estimated_batches = (total_questions + batch_size - 1) // batch_size
    print(f"使用批处理大小 {batch_size}，预计将处理 {estimated_batches} 个批次")

    # 记录开始时间
    start_time = time.time()

    print(f"开始生成数据集，处理 {len(questions)} 个问题...")
    # 使用优化版本的生成函数
    thinking_pools, error_thinking_pools, selected_results = generate_dataset_batched(  # 更新为接收三个返回值
        model=model,
        questions=questions,
        num_iterations=8,  # 最多3个步骤
        alpha=1.0,
        beta=1.0,
        batch_size=batch_size,  # 使用设定的批处理大小
        tokenizer=tokenizer
    )

    # 记录生成数据集的结束时间
    generation_end_time = time.time()
    generation_duration = generation_end_time - start_time
    print(f"数据集生成完成，耗时: {generation_duration:.2f} 秒（{generation_duration / 60:.2f} 分钟）")

    print(f"保存正确思维池结果到 {thinking_pools_path}...")
    with open(thinking_pools_path, 'w', encoding='utf-8') as f:
        json.dump(thinking_pools, f, ensure_ascii=False, indent=2)

    print(f"保存错误思维池结果到 {error_thinking_pools_path}...")  # 新增：保存错误思维池
    with open(error_thinking_pools_path, 'w', encoding='utf-8') as f:
        json.dump(error_thinking_pools, f, ensure_ascii=False, indent=2)

    print(f"保存选择的步骤结果到 {selected_results_path}...")
    with open(selected_results_path, 'w', encoding='utf-8') as f:
        json.dump(selected_results, f, ensure_ascii=False, indent=2)

    # 选择最优解并保存
    optimal_solutions = select_optimal_solutions(thinking_pools, selected_results)

    # 记录选择最优解的结束时间
    selection_end_time = time.time()
    selection_duration = selection_end_time - generation_end_time

    print(f"保存最优解结果到 {optimal_solutions_path}...")
    with open(optimal_solutions_path, 'w', encoding='utf-8') as f:
        json.dump(optimal_solutions, f, ensure_ascii=False, indent=2)

    # 分析结果
    analyze_results(optimal_solutions_path)

    # 新增：分析错误思维池
    analyze_error_thinking_pools(error_thinking_pools_path)

    # 记录总时间
    total_duration = time.time() - start_time
    print(
        f"所有处理完成，总耗时: {total_duration:.2f} 秒（{total_duration / 60:.2f} 分钟，{total_duration / 3600:.2f} 小时）")
    print("完成！")