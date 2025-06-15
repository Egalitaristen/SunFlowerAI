import json
import re
from datasets import load_dataset

def parse_integer_answer(text: str) -> int | None:
    """Extracts the an integer from the model's response.
    Attempts to find numbers and assumes the last one is the answer.
    AIME answers are integers between 0 and 999.
    """
    if not isinstance(text, str):
        return None

    # Find all sequences of digits
    numbers = re.findall(r'\d+', text)

    if numbers:
        # Try to convert the last found number to an integer
        try:
            # AIME answers are 0-999. This is a simple check.
            # More sophisticated parsing might be needed if model output is complex.
            ans = int(numbers[-1])
            # if 0 <= ans <= 999:
            #     return ans
            return ans # Keep it simple, allow any integer for now, can be filtered later if needed
        except ValueError:
            return None
    return None

def evaluate_aime(results_filepath: str, ground_truth_dataset_name: str = "opencompass/AIME2025", ground_truth_split: str = "test"):
    """Evaluates AIME benchmark results."""
    print(f"Starting AIME evaluation for: {results_filepath}")

    # Load the ground truth answers
    ground_truth = {}
    try:
        dataset = load_dataset(ground_truth_dataset_name, "default", split=ground_truth_split)
        for item in dataset:
            if 'question' in item and 'answer' in item:
                # Ensure answer is an int for comparison
                try:
                    ground_truth[item['question']] = int(item['answer'])
                except (ValueError, TypeError):
                    print(f"Warning: Could not parse ground truth answer for question: {item['question'][:50]}... Answer: {item['answer']}")
            else:
                print(f"Warning: Skipping ground truth item due to missing 'question' or 'answer': {item}")
        print(f"Loaded {len(ground_truth)} ground truth AIME questions.")
    except Exception as e:
        print(f"Error loading AIME ground truth dataset '{ground_truth_dataset_name}': {e}. Please ensure 'datasets' is installed and you have internet access.")
        return

    if not ground_truth:
        print("No ground truth data loaded. Cannot proceed with evaluation.")
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
        print("No agent results loaded. Cannot proceed with evaluation.")
        return

    correct_count = 0
    total_count = 0

    for result in agent_results:
        question = result.get('input')
        # We evaluate the 'prompted_response' as per the plan
        agent_response_text = result.get('prompted_response')

        if question is None or agent_response_text is None:
            print(f"Warning: Skipping agent result due to missing 'input' or 'prompted_response': {result}")
            continue

        total_count += 1
        agent_answer_parsed = parse_integer_answer(agent_response_text)
        correct_answer_gt = ground_truth.get(question)

        print(f"\nQ: {question[:70].strip()}...")
        print(f"  - Ground Truth Answer: {correct_answer_gt}")
        # print(f"  - Agent's Raw Response: {str(agent_response_text)[:100].strip()}...")
        print(f"  - Parsed Agent Answer: {agent_answer_parsed}")

        if correct_answer_gt is not None:
            if agent_answer_parsed == correct_answer_gt:
                correct_count += 1
                print("  - Result: CORRECT")
            else:
                print("  - Result: INCORRECT")
        else:
            print(f"  - Result: SKIPPED (No ground truth found for this question)")

    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print("\n" + "="*30)
        print("AIME Benchmark Results Summary")
        print("="*30)
        print(f"Total Questions Processed: {total_count}")
        print(f"Correct Agent Answers:   {correct_count}")
        print(f"Accuracy:                {accuracy:.2f}%")
    else:
        print("\nNo questions were processed from the agent's results.")

if __name__ == "__main__":
    # Ensure necessary libraries are installed
    try:
        import datasets
    except ImportError as e:
        print(f"ImportError: {e}. Please install the required libraries. You might need to run: pip install datasets")
    else:
        # This assumes aime_results.json is in the same directory as this script
        evaluate_aime("aime_results.json")
