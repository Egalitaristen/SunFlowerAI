import json
import re
from datasets import load_dataset
import config  # Import our configuration file

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

def evaluate_hle(results_filepath: str = None):
    """Evaluates Humanity's Last Exam benchmark results."""
    # Use config defaults if not provided
    results_filepath = results_filepath or config.HLE_RESULTS_FILE
    
    print(f"Starting HLE evaluation for: {results_filepath}")

    # Load the ground truth data from Hugging Face
    ground_truth = {}
    try:
        # Load HLE dataset from Hugging Face with token if provided
        if config.HF_TOKEN and config.HF_TOKEN != "YOUR_HF_TOKEN":
            dataset = load_dataset("cais/hle", split="test", token=config.HF_TOKEN)
        else:
            dataset = load_dataset("cais/hle", split="test")
        print("Loading HLE ground truth from Hugging Face...")
        
        # Handle both IterableDataset and regular Dataset
        if hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__'):
            # IterableDataset - iterate directly
            for item in dataset:
                if 'question' in item and 'answer' in item and 'question_type' in item:
                    ground_truth[item['question']] = {
                        'answer': item['answer'],
                        'type': item['question_type']
                    }
                else:
                    print(f"Warning: Skipping HLE ground truth item due to missing fields: {item}")
        else:
            # Regular Dataset - use indexing
            try:
                dataset_length = len(dataset)
                for i in range(dataset_length):
                    item = dataset[i]
                    if 'question' in item and 'answer' in item and 'question_type' in item:
                        ground_truth[item['question']] = {
                            'answer': item['answer'],
                            'type': item['question_type']
                        }
                    else:
                        print(f"Warning: Skipping HLE ground truth item due to missing fields: {item}")
            except (TypeError, AttributeError):
                # Fallback if len() doesn't work - treat as iterable
                for item in dataset:
                    if 'question' in item and 'answer' in item and 'question_type' in item:
                        ground_truth[item['question']] = {
                            'answer': item['answer'],
                            'type': item['question_type']
                        }
                    else:
                        print(f"Warning: Skipping HLE ground truth item due to missing fields: {item}")
        
        print(f"Loaded {len(ground_truth)} ground truth HLE questions from Hugging Face.")
    except Exception as e:
        if "gated dataset" in str(e) or "authenticated" in str(e):
            print(f"Error: HLE dataset access denied. The HLE dataset on Hugging Face is gated and requires authentication.")
            print("To access HLE dataset:")
            print("1. Create a Hugging Face account at https://huggingface.co/")
            print("2. Request access to the dataset at https://huggingface.co/datasets/cais/hle")
            print("3. Get your token from https://huggingface.co/settings/tokens")
            print("4. Add your token to config.py: HF_TOKEN = 'your_token_here'")
        elif "Pillow" in str(e) or "PIL" in str(e):
            print(f"Error: Missing required dependency for image processing.")
            print("The HLE dataset contains multimodal questions that require image processing.")
            print("Please install Pillow: pip install pillow")
            print("Then try running the evaluation again.")
        else:
            print(f"Error loading HLE dataset from Hugging Face: {e}. Please ensure 'datasets' library is installed and you have internet connectivity.")
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
    # Use config file for default filenames
    try:
        import config
    except ImportError:
        print("Error: config.py not found. Make sure config.py is in the same directory as this script.")
    else:
        evaluate_hle()
