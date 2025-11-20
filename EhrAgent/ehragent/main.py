import os
import json
import random
import numpy as np
import argparse
import autogen
from toolset_high import *
from medagent import MedAgent
from config import openai_config, llm_config_list
import time
import traceback
from execution import LOG_SEPARATOR, execute_with_memory

def judge(pred, ans):
    old_flag = True
    if not ans in pred:
        old_flag = False
    if "True" in pred:
        pred = pred.replace("True", "1")
    else:
        pred = pred.replace("False", "0")
    if ans == "False" or ans == "false":
        ans = "0"
    if ans == "True" or ans == "true":
        ans = "1"
    if ans == "No" or ans == "no":
        ans = "0"
    if ans == "Yes" or ans == "yes":
        ans = "1"
    if ans == "None" or ans == "none":
        ans = "0"
    if ", " in ans:
        ans = ans.split(', ')
    if ans[-2:] == ".0":
        ans = ans[:-2]
    if not type(ans) == list:
        ans = [ans]
    new_flag = True
    for i in range(len(ans)):
        if not ans[i] in pred:
            new_flag = False
            break
    return (old_flag or new_flag)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default=os.getenv("LLM_MODEL"))
    parser.add_argument("--tracllm_config", type=str, default=os.getenv("TRACLLM_MODEL_CONFIG"))
    parser.add_argument("--tracllm_device", type=str, default=os.getenv("TRACLLM_DEVICE", "cpu"))
    parser.add_argument("--tracllm_top_k", type=int, default=int(os.getenv("TRACLLM_TOP_K", 5)))
    parser.add_argument("--tracllm_explanation_level", type=str, default=os.getenv("TRACLLM_EXPLANATION_LEVEL", "sentence"))
    parser.add_argument("--tracllm_score_funcs", type=str, default=os.getenv("TRACLLM_SCORE_FUNCS", "stc,loo,denoised_shapley"))
    parser.add_argument("--tracllm_shapley_samples", type=int, default=int(os.getenv("TRACLLM_SHAPLEY_SAMPLES", 5)))
    parser.add_argument("--tracllm_beta", type=float, default=float(os.getenv("TRACLLM_BETA", 0.2)))
    parser.add_argument("--tracllm_loo_weight", type=float, default=float(os.getenv("TRACLLM_LOO_WEIGHT", 2)))
    parser.add_argument("--tracllm_verbose", type=int, default=int(os.getenv("TRACLLM_VERBOSE", 1)))
    parser.add_argument("--num_questions", type=int, default=int(os.getenv("NUM_QUESTIONS")))
    parser.add_argument("--dataset", type=str, default=os.getenv("DATASET"))
    parser.add_argument("--data_path", type=str, default=os.getenv("DATASET_PATH"))
    parser.add_argument("--logs_path", type=str, default=os.getenv("LOGS_PATH"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_id", type=str, default="521fd2885f51641a963f8d3e")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--num_shots", type=int, default=int(os.getenv("NUM_SHOTS")))
    parser.add_argument("--use_memsize_after", type=int, default=int(os.getenv("USE_MEMORY_SIZE_AFTER")))
    args = parser.parse_args()

    tracllm_score_funcs = [part.strip() for part in str(args.tracllm_score_funcs).split(",") if part.strip()]

    set_seed(args.seed)

    if args.dataset == 'mimic_iii':
        from prompts_mimic import EHRAgent_4Shots_Knowledge
    else:
        from prompts_eicu import EHRAgent_4Shots_Knowledge

    config_list = [openai_config(args.llm)]
    llm_config = llm_config_list(args.seed, config_list)

    chatbot = autogen.agentchat.AssistantAgent(
        name="chatbot",
        system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done. Save the answers to the questions in the variable 'answer'. Please only generate the code.",
        llm_config=llm_config,
    )

    user_proxy = MedAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config={
            "work_dir": "coding",
            "use_docker": False,
        },
        config_list=config_list,
    )

    # register the functions
    user_proxy.register_function(
        function_map={
            "python": run_code
        }
    )

    user_proxy.register_dataset(args.dataset)

    file_path = f"{args.data_path}/{args.dataset}/valid_preprocessed.json"
    # read from json file
    with open(file_path, 'r') as f:
        contents = json.load(f)

    # random shuffle
    random.shuffle(contents)
    log_file_path = "{}/{}/log/".format(args.logs_path, args.num_shots)+"{id}.txt"
    sol_file_path = "{}/{}/sol/".format(args.logs_path, args.num_shots)+"{id}_ans.txt"

    start_time = time.time()
    if args.num_questions == -1:
        args.num_questions = len(contents)

    long_term_memory = []
    init_memory = EHRAgent_4Shots_Knowledge
    init_memory = init_memory.split('\n\n')
    for i in range(len(init_memory)):
        item = init_memory[i]
        item = item.split('Question:')[-1]
        question = item.split('\nKnowledge:\n')[0]
        item = item.split('\nKnowledge:\n')[-1]
        knowledge = item.split('\nSolution:')[0]
        code = item.split('\nSolution:')[-1]
        new_item = {"question": question, "knowledge": knowledge, "code": code}
        long_term_memory.append(new_item)

    tracllm_wrapper = None

    stats = {
        "total": 0,
        "correct": 0,
        "incorrect": 0,
        "unfinished": 0,
        "tracllm_used": 0,
        "tracllm_improved": 0,
        "memory_size": len(long_term_memory)
    }

    for i in range(args.start_id, args.num_questions):
        if args.debug and contents[i]['id'] != args.debug_id:
            continue

        use_tracllm = len(long_term_memory) > args.use_memsize_after

        print(f"\n{'='*60}")
        print(f"Processing question {i+1}/{args.num_questions}")
        print(f"ID: {contents[i]['id']}")
        print(f"Current memory size: {len(long_term_memory)}")
        print(f"TracLLM mode: {'ENABLED' if use_tracllm else 'DISABLED'} (threshold: {args.use_memsize_after})")
        print(f"{'='*60}\n")

        #print(contents[i])
        question = contents[i]['template']
        answer = contents[i]['answer']

        logs_string = []
        autogen_log = ""
        is_correct = False

        execution_failed = False

        try:
            if not use_tracllm:
                print("Using Levenshtein distance for memory retrieval....")
                logs_string, autogen_log = execute_with_memory(
                    user_proxy,
                    chatbot,
                    question,
                    answer,
                    long_term_memory,
                    num_shots=args.num_shots,
                )

            else:
                print("Using TracLLM-based memory retrieval...")
                stats["tracllm_used"] += 1

                if tracllm_wrapper is None:
                    from tracllm_wrapper import TracLLMWrapper
                    tracllm_wrapper = TracLLMWrapper(
                        config_path=args.tracllm_config,
                        device=args.tracllm_device,
                        explanation_level=args.tracllm_explanation_level,
                        top_k=args.tracllm_top_k,
                        score_funcs=tracllm_score_funcs,
                        shapley_samples=args.tracllm_shapley_samples,
                        loo_weight=args.tracllm_loo_weight,
                        beta=args.tracllm_beta,
                        verbose=args.tracllm_verbose
                    )
                    print("TracLLM wrapper initialized")

                # Step 1: Get top-k memory indices using Levenshtein
                top_k_indices = user_proxy.get_top_k_memory_indices(
                    question,
                    long_term_memory,
                    args.num_shots
                )

                # Get next condidates (k+1, k+2, ...) for potential replacement
                all_indices_sorted = user_proxy.get_all_memory_indices_sorted(question, long_term_memory)
                candidate_indices = [idx for idx in all_indices_sorted if idx not in top_k_indices]

                # step 2: Execute with initial top-k
                selected_memory = [long_term_memory[idx] for idx in top_k_indices]
                logs_string, autogen_log = execute_with_memory(
                    user_proxy,
                    chatbot,
                    question,
                    answer,
                    selected_memory
                )

                # Step 3: Check if execution is correct
                logs_string_joined = LOG_SEPARATOR.join(logs_string)

                if "TERMINATE" in logs_string_joined:
                    # Extract prediction
                    if '"cell": "' in logs_string_joined:
                        last_code_end = logs_string_joined.rfind('"\n}')
                    else:
                        last_code_end = logs_string_joined.rfind('Solution:')
                    prediction_end = logs_string_joined.rfind('TERMINATE')
                    prediction = logs_string_joined[last_code_end:prediction_end]

                    if type(answer) == list:
                        answer_str = ', '.join(answer)
                    else:
                        answer_str = str(answer)

                    is_correct = judge(prediction, answer_str)

                    # Step 4: If incorrect, use TracLLM attribution
                    if not is_correct and len(candidate_indices) > 0:
                        print("Initial execution incorrect. Running TracLLM attribution...")

                        # Get attribution scores
                        ranked_records = tracllm_wrapper.score_memory(
                            query=question,
                            response=autogen_log,
                            recs=selected_memory
                        )

                        print(f"Attribution scores (highest to lowest):")
                        for score, rec, rec_idx in ranked_records:
                            print(f"Score: {score:.4f} - Question: {rec['question'][:50]}...")

                        # Find most problematic memory
                        highest_score, problematic_record, problematic_idx_in_topk = ranked_records[0]

                        if problematic_idx_in_topk is not None:
                            problematic_memory_idx = top_k_indices[problematic_idx_in_topk]
                            print(f"Most problematic memory index: {problematic_memory_idx} (Score: {highest_score:.4f})")

                            # Try replacing with next candidates
                            max_replacements = min(10, len(candidate_indices))

                            for replacement_attempt in range(max_replacements):
                                if replacement_attempt >= len(candidate_indices):
                                    break

                                replacement_idx = candidate_indices[replacement_attempt]
                                print(f"\nReplacement attempt {replacement_attempt+1}: Replacing index {problematic_memory_idx} with {replacement_idx}")

                                # Create new memory selection
                                new_selected_memory = selected_memory.copy()
                                new_selected_memory[problematic_idx_in_topk] = long_term_memory[replacement_idx]

                                # Re-execute
                                new_logs_string, new_autogen_log = execute_with_memory(
                                    user_proxy,
                                    chatbot,
                                    question,
                                    answer,
                                    new_selected_memory
                                )

                                # Check new result
                                new_logs_joined = LOG_SEPARATOR.join(new_logs_string)

                                if "TERMINATE" in new_logs_joined:
                                    if '"cell": "' in new_logs_joined:
                                        last_code_end = new_logs_joined.rfind('"\n}')
                                    else:
                                        last_code_end = new_logs_joined.rfind('Solution:')
                                    
                                    prediction_end = new_logs_joined.rfind('TERMINATE')
                                    new_prediction = new_logs_joined[last_code_end:prediction_end]

                                    new_is_correct = judge(new_prediction, answer_str)

                                    if new_is_correct:
                                        print("Replacement successful! New execution is correct.")
                                        logs_string = new_logs_string
                                        autogen_log = new_autogen_log
                                        is_correct = True
                                        stats["tracllm_improved"] += 1
                                        break
                                    else:
                                        print(f"Replacement attempt {replacement_attempt+1} still incorrect.")

                                else:
                                    print(f"Replacement attempt {replacement_attempt+1} did not terminate properly.")

                            if not is_correct:
                                print("All replacement attempts exhausted. Execution remains incorrect.")

        except Exception as e:
            print(f"Exception during execution: {e}")
            traceback.print_exc()
            # Preserve existing logs if any
            if not logs_string:
                logs_string = [str(question), str(answer)]
            logs_string.append(f"EXECUTION ERROR: {str(e)}")
            execution_failed = True
        
        # Prepare answer for logging
        if type(answer) == list:
            answer = ', '.join(answer)
        logs_string.append("Ground-Truth Answer ---> "+answer)

        # Define file paths
        log_directory = log_file_path.format(id=contents[i]['id'])
        sol_directory = sol_file_path.format(id=contents[i]['id'])
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_directory), exist_ok=True)
        os.makedirs(os.path.dirname(sol_directory), exist_ok=True)

        # Write logs to file
        with open(log_directory, 'w') as f:
            f.write(LOG_SEPARATOR.join(logs_string))
        logs_string_joined = LOG_SEPARATOR.join(logs_string)

        stats["total"] += 1

        if execution_failed:
            stats["unfinished"] += 1
            print("Execution error - not adding to memory")
            with open(sol_directory, 'w') as f:
                f.write("Execution ERROR - not adding to memory")

        elif "TERMINATE" in logs_string_joined:
            if '"cell"' in logs_string_joined:
                code_end_token = '"}\n'
                last_code_end = logs_string_joined.rfind(code_end_token)
                if last_code_end != -1:
                    last_code_end += len(code_end_token)
                else:
                    # Fallback to the start of the last executed cell
                    last_code_end = logs_string_joined.rfind('{"cell"')
            else:
                last_code_end = logs_string_joined.rfind('Solution:')

            prediction_end = logs_string_joined.rfind("TERMINATE")
            prediction = logs_string_joined[last_code_end:prediction_end]

            # Clean prediction for solution file

            prediction_cleaned = prediction.replace(LOG_SEPARATOR.strip("\n"), '')
            prediction_cleaned = prediction_cleaned.replace('"}\n', '')
            prediction_cleaned = '\n'.join(line.strip() for line in prediction_cleaned.split('\n') if line.strip())

            # Write to solution file
            with open(sol_directory, 'w') as f:
                f.write(prediction_cleaned)

            if not use_tracllm:
                result = judge(prediction, answer)
            else:
                result = is_correct

            if result:
                stats["correct"] += 1
                print("Correct answer!")

                new_item = {
                    "question": question,
                    "knowledge": user_proxy.knowledge,
                    "code": user_proxy.code,
                    "id": contents[i]['id']
                }

                long_term_memory.append(new_item)
                print(f"Added to memory. New memory size: {len(long_term_memory)}")

            else:
                stats["incorrect"] += 1
                print("Incorrect answer.")
                print(f"Expected: {answer}")
                print(f"Got: {prediction_cleaned}")

        else:
            # No TERMINATE found
            stats["unfinished"] += 1
            print("Conversation did not terminate properly")
            with open(sol_directory, 'w') as f:
                f.write("No TERMINATE found - not adding to memory")

        # Update and print stats
        stats["memory_size"] = len(long_term_memory)

        print(f"\nRunning Stats:")
        print(f"  Total: {stats['total']}")
        print(f"  Correct: {stats['correct']} ({stats['correct']/stats['total']*100:.1f}%)")
        print(f"  Incorrect: {stats['incorrect']}")
        print(f"  Unfinished: {stats['unfinished']}")
        if stats["tracllm_used"] > 0:
            print(f"  TracLLM Used: {stats['tracllm_used']}")
            print(f"  TracLLM Improved: {stats['tracllm_improved']}")
        print(f"  Memory Bank: {stats['memory_size']}")

    end_time = time.time()

    # Final statistics
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total questions: {stats['total']}")
    print(f"Correct: {stats['correct']} ({stats['correct']/stats['total']*100:.2f}%)")
    print(f"Incorrect: {stats['incorrect']}")
    print(f"Unfinished: {stats['unfinished']}")
    if stats["tracllm_used"] > 0:
        print(f"TracLLM Used: {stats['tracllm_used']}")
        print(f"TracLLM Improved: {stats['tracllm_improved']} ({stats['tracllm_improved']/stats['tracllm_used']*100:.1f}%)")
    print(f"Final memory bank size: {stats['memory_size']}")
    print(f"Time elapsed: {end_time - start_time:.2f} seconds")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
