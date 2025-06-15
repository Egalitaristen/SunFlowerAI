import json
import re

def get_multiple_choice_answer(text: str) -> str | None:
    """Extracts a single uppercase letter, assumed to be the answer for MCQs.
    Looks for common patterns like 'Answer: A', 'is A)', or a standalone letter.
    """
    if not isinstance(text, str):
        return None

    # Pattern 1: Explicitly stated answer, e.g., "Answer: A", "The answer is A."
    # Handles variations in spacing and casing of "answer is/answer:".
    match = re.search(r'(?:[Aa]nswer(?:\s*is|\s*:)?)\s*([A-Z])(?:[\.\s\,\)]|$)', text)
    if match:
        return match.group(1).upper()

    # Pattern 2: Letter enclosed in parentheses, e.g., "(A)"
    match = re.search(r'\(([A-Z])\)', text)
    if match:
        return match.group(1).upper()

    # Pattern 3: A standalone capital letter, possibly followed by a period or at line end.
    # This is a bit more general and should be applied carefully.
    # Try to find it at the end of the string or a line, or surrounded by spaces.
    match = re.search(r'(?:^|\s)([A-Z])(?:[\.\s]|$)', text)
    if match:
        return match.group(1).upper()

    # Fallback: if the response is just a single letter
    if len(text.strip()) == 1 and text.strip().isalpha():
        return text.strip().upper()

    return None

def evaluate_hle(results_filepath: str, ground_truth_filepath: str):
    """Evaluates Humanity's Last Exam benchmark results."""
    print(f"Starting HLE evaluation for: {results_filepath} using ground truth: {ground_truth_filepath}")

    # Load the ground truth data
    ground_truth = {}
    try:
        with open(ground_truth_filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    if 'question' in data and 'answer' in data and 'question_type' in data:
                        ground_truth[data['question']] = {
                            'answer': data['answer'],
                            'type': data['question_type']
                        }
                    else:
                        print(f"Warning: Skipping HLE ground truth line {line_num} due to missing fields: {line.strip()}")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON in HLE ground truth line {line_num}: {line.strip()}")
        print(f"Loaded {len(ground_truth)} ground truth HLE questions from {ground_truth_filepath}.")
    except FileNotFoundError:
        print(f"Error: HLE ground truth file not found: {ground_truth_filepath}. Please ensure you have cloned the HLE repository and the path is correct.")
        return

    if not ground_truth:
        print("No ground truth data loaded for HLE. Cannot proceed.")
        return

    # Load your agent's results
    try:
        with open(results_filepath, 'r', encoding='utf-8') as f:
            agent_results = json.load(f)
        print(f"Loaded {len(agent_results)} agent results from {results_filepath}.")
    except FileNotFoundError:
        print(f"Error: Agent results file not found: {results_filepath}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from agent results file: {results_filepath}")
        return

    if not agent_results:
        print("No agent results loaded for HLE. Cannot proceed.")
        return

    mc_correct, fr_correct = 0, 0
    mc_total, fr_total = 0, 0
    unmatched_gt_questions = 0

    for result_idx, result in enumerate(agent_results, 1):
        question = result.get('input')
        agent_response = result.get('prompted_response') # Evaluating 'prompted_response'

        if question is None or agent_response is None:
            print(f"Warning: Skipping agent result item {result_idx} due to missing 'input' or 'prompted_response'.")
            continue

        if question not in ground_truth:
            # print(f"Warning: Question from agent results not found in HLE ground truth: {question[:50]}...")
            unmatched_gt_questions +=1
            continue

        gt_data = ground_truth[question]
        gt_answer_text = str(gt_data['answer']) # Ensure ground truth answer is a string
        q_type = gt_data['type']

        # print(f"\nQ: {question[:70].strip()}...")
        # print(f"  - Type: {q_type}, GT Answer: {gt_answer_text}")
        # print(f"  - Agent Raw: {str(agent_response)[:100].strip()}...")

        if q_type == 'multiple_choice':
            mc_total += 1
            agent_mc_answer = get_multiple_choice_answer(str(agent_response))
            # print(f"  - Parsed MC Agent Answer: {agent_mc_answer}")
            if agent_mc_answer and agent_mc_answer == gt_answer_text.upper():
                mc_correct += 1
                # print("  - Result: CORRECT (MC)")
            # else:
                # print("  - Result: INCORRECT (MC)")
        elif q_type == 'free_response':
            fr_total += 1
            # Simple check: case-insensitive substring match of the ground truth string
            # More sophisticated free-text evaluation (e.g., ROUGE, BLEU, semantic similarity) is complex.
            if gt_answer_text.lower() in str(agent_response).lower():
                fr_correct += 1
                # print("  - Result: CORRECT (FR)")
            # else:
                # print("  - Result: INCORRECT (FR)")
        else:
            print(f"Warning: Unknown question type '{q_type}' for question: {question[:50]}...")

    if unmatched_gt_questions > 0:
        print(f"\nWarning: {unmatched_gt_questions} questions from agent results were not found in the HLE ground truth set and were skipped.")

    # Calculate scores
    mc_accuracy = (mc_correct / mc_total * 100) if mc_total > 0 else 0
    fr_accuracy = (fr_correct / fr_total * 100) if fr_total > 0 else 0

    total_processed_questions = mc_total + fr_total
    total_overall_correct = mc_correct + fr_correct
    overall_accuracy = (total_overall_correct / total_processed_questions * 100) if total_processed_questions > 0 else 0

    print("\n" + "="*30)
    print("Humanity's Last Exam Results Summary")
    print("="*30)
    print(f"Multiple Choice Processed: {mc_total}")
    print(f"Multiple Choice Correct:   {mc_correct} ({mc_accuracy:.2f}%)")
    print("-"*30)
    print(f"Free Response Processed:   {fr_total}")
    print(f"Free Response Correct:     {fr_correct} ({fr_accuracy:.2f}%)")
    print("-"*30)
    print(f"Total Questions Evaluated: {total_processed_questions}")
    print(f"Overall Correct Answers:   {total_overall_correct}")
    print(f"Overall Accuracy:          {overall_accuracy:.2f}%")
    print("="*30)
    if total_processed_questions < len(agent_results):
        print(f"Note: {len(agent_results) - total_processed_questions} results were not evaluated (e.g. missing in ground truth or malformed). Check warnings above.")


if __name__ == "__main__":
    # This assumes hle_results.json is in the same directory
    # and the HLE data is in 'hle/data/hle_test_set.jsonl'
    results_file = "hle_results.json"
    # Path to HLE data as specified in run_my_agent.py and user guide
    ground_truth_file = "hle/data/hle_test_set.jsonl"

    evaluate_hle(results_file, ground_truth_file)
