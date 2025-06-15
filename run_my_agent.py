import asyncio
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time
import argparse
from groq import AsyncGroq
from datasets import load_dataset

@dataclass
class ComparisonResult:
    input_text: str
    direct_response: str
    generated_system_prompt: str
    prompted_response: str

class DynamicPromptComparison:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192",
                 rate_limit_delay: float = 0.5):
        """
        Initialize with Groq API client

        Args:
            api_key: Groq API key
            model: Model name to use (default: llama3-8b-8192)
            rate_limit_delay: Delay between API calls in seconds
        """
        self.client = AsyncGroq(api_key=api_key)
        self.model = model
        self.rate_limit_delay = rate_limit_delay

        # Your refined system prompt generator prompt
        self.generator_prompt = """You are an expert system prompt generator for AI models. Your task is to create tailored, comprehensive instructions that guide AI assistants in responding effectively to various contexts and queries. When presented with a scenario or task, carefully analyze all aspects of the given context. Consider the overall goal, specific requirements, and any constraints or preferences mentioned. Determine the appropriate tone, level of formality, and target audience for the response. Begin your system prompt by clearly defining the AI's role with "You are an expert [relevant role]" to establish the appropriate context and expertise. Provide clear, concise instructions tailored to the specific scenario, maintaining an objective and impartial tone throughout. Encourage structured thinking by incorporating phrases like "Think carefully and step-by-step" to prompt the AI to break down its approach into clear stages. Outline the necessary methodology without including full response content. Focus solely on guiding the AI's approach and response structure. Craft your instructions in flowing paragraphs rather than numbered or bulleted lists to avoid inadvertently influencing the AI's output format. Ensure your guidance allows for natural and appropriate responses to the given task or query. Consider potential limitations of the AI model you're instructing. Assume it may not have access to human feedback, internet connectivity, or the ability to generate non-text outputs. Instruct the AI to rely only on its internal knowledge and capabilities. Thoroughly review your generated prompt to ensure it comprehensively addresses all aspects of the given context while maintaining a focus on guiding the AI's approach. Verify that your instructions are clear, coherent, and avoid any elements that could lead to inappropriate or misaligned responses. Remember, your output should consist solely of the system prompt itself, without additional explanations or examples. Refrain from including full response content such as complete recipes, jokes, or stories. Your goal is to provide a framework for the AI to generate its own appropriate and contextually relevant responses. Unless the task is for programming, output the prompt in the same language as the input."""

    async def _make_api_call(self, system_prompt: str, user_message: str) -> str:
        """Make API call with rate limiting"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            return response.choices[0].message.content
        except Exception as e:
            print(f"An API error occurred: {e}")
            # Fallback for rate limit or other errors to avoid script halt
            await asyncio.sleep(self.rate_limit_delay * 5) # Longer delay on error
            return f"Error: API call failed - {e}" # Return error message

    async def generate_system_prompt(self, user_input: str) -> str:
        """Generate a dynamic system prompt for the given input"""
        return await self._make_api_call(
            self.generator_prompt,
            f"Generate a system prompt for this input: {user_input}"
        )

    async def get_direct_response(self, user_input: str) -> str:
        """Get response without any system prompt"""
        return await self._make_api_call("", user_input)

    async def get_prompted_response(self, system_prompt: str, user_input: str) -> str:
        """Get response with the generated system prompt"""
        return await self._make_api_call(system_prompt, user_input)

    async def compare_single_input(self, user_input: str) -> ComparisonResult:
        """Run comparison for a single input"""
        print(f"Processing: {user_input[:60]}...")

        direct_response = await self.get_direct_response(user_input)
        generated_prompt = await self.generate_system_prompt(user_input)
        prompted_response = await self.get_prompted_response(generated_prompt, user_input)

        return ComparisonResult(
            input_text=user_input,
            direct_response=direct_response,
            generated_system_prompt=generated_prompt,
            prompted_response=prompted_response
        )

    async def compare_batch(self, inputs: List[str]) -> List[ComparisonResult]:
        """Run comparison for multiple inputs"""
        results = []
        total = len(inputs)

        for i, user_input in enumerate(inputs, 1):
            print(f"Progress: {i}/{total}")
            # Basic check to ensure input is a string
            if not isinstance(user_input, str):
                print(f"Skipping invalid input (not a string): {user_input}")
                results.append(ComparisonResult(
                    input_text=str(user_input), # Store string representation
                    direct_response="Error: Invalid input type",
                    generated_system_prompt="Error: Invalid input type",
                    prompted_response="Error: Invalid input type"
                ))
                continue
            result = await self.compare_single_input(user_input)
            results.append(result)

        return results

    def print_comparison(self, result: ComparisonResult) -> None:
        """Print formatted comparison for a single result"""
        print("\n" + "="*80)
        print("INPUT:")
        print(result.input_text)
        print("\n" + "-"*40)
        print("GENERATED SYSTEM PROMPT:")
        print(result.generated_system_prompt)
        print("\n" + "-"*40)
        print("DIRECT RESPONSE (no system prompt):")
        print(result.direct_response)
        print("\n" + "-"*40)
        print("PROMPTED RESPONSE (with generated system prompt):")
        print(result.prompted_response)
        print("="*80)

    def save_results(self, results: List[ComparisonResult], filename: str = "comparison_results.json") -> None:
        """Save results to JSON file"""
        data = []
        for result in results:
            data.append({
                "input": result.input_text,
                "direct_response": result.direct_response,
                "generated_system_prompt": result.generated_system_prompt,
                "prompted_response": result.prompted_response
            })

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to {filename}")

def load_hle_questions(filepath: str, max_questions: int | None = None) -> List[str]:
    """Loads questions from the HLE JSONL file, with an optional limit."""
    questions = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_questions and i >= max_questions:
                    print(f"Loaded a maximum of {max_questions} questions as requested.")
                    break
                try:
                    data = json.loads(line)
                    questions.append(data['question'])
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line in {filepath}: {line.strip()}")
                except KeyError:
                    print(f"Warning: Skipping line in {filepath} missing 'question' field: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: HLE questions file not found at {filepath}. Please ensure you have cloned the HLE repository (git clone https://github.com/centerforaisafety/hle.git) and the path is correct.")
        return []

    if max_questions and len(questions) == max_questions:
        # This condition is met if loop broke due to max_questions
        pass # Message already printed
    elif max_questions and len(questions) < max_questions:
         print(f"Loaded {len(questions)} questions from HLE file (less than max_questions limit of {max_questions}): {filepath}")
    else:
        print(f"Loaded {len(questions)} questions from HLE file: {filepath}")
    return questions

def load_aime_questions(max_questions: int | None = None) -> List[str]:
    """Loads questions from the AIME Hugging Face dataset, with an optional limit."""
    questions = []
    try:
        dataset = load_dataset("opencompass/AIME2025", "default", split="test")
        
        # Handle both IterableDataset and regular Dataset
        if hasattr(dataset, '__iter__') and not hasattr(dataset, '__getitem__'):
            # IterableDataset - iterate directly
            for i, item in enumerate(dataset):
                if max_questions and i >= max_questions:
                    print(f"Loaded a maximum of {max_questions} questions as requested.")
                    break
                if 'question' in item and isinstance(item['question'], str):
                    questions.append(item['question'])
                else:
                    print(f"Warning: Skipping AIME item due to missing or invalid 'question' field: {item}")
        else:
            # Regular Dataset - use indexing
            total_items = len(dataset) if hasattr(dataset, '__len__') else float('inf')
            max_to_process = min(max_questions or total_items, total_items)
            
            for i in range(int(max_to_process)):
                item = dataset[i]
                if 'question' in item and isinstance(item['question'], str):
                    questions.append(item['question'])
                else:
                    print(f"Warning: Skipping AIME item due to missing or invalid 'question' field: {item}")

        if max_questions and len(questions) == max_questions:
            pass # Message already printed
        elif max_questions and len(questions) < max_questions:
            print(f"Loaded {len(questions)} questions from AIME dataset (less than max_questions limit of {max_questions}).")
        else:
            print(f"Loaded {len(questions)} questions from AIME dataset.")

    except Exception as e:
        print(f"Error loading AIME dataset: {e}. Please ensure 'datasets' library is installed and you have internet connectivity.")
        return []
    return questions

async def main():
    parser = argparse.ArgumentParser(description="Run dynamic prompt comparison with Groq API on HLE or AIME benchmarks.")
    parser.add_argument(
        "--max_questions",  # Fixed: using underscore to match the variable name
        type=int,
        default=None,
        help="Maximum number of questions to process from the benchmark."
    )
    args = parser.parse_args()

    # IMPORTANT: Replace with your actual Groq API key
    api_key = "YOUR_GROQ_API_KEY"

    if api_key == "YOUR_GROQ_API_KEY":
        print("ERROR: Please replace 'YOUR_GROQ_API_KEY' in run_my_agent.py with your actual Groq API key.")
        return

    comparison = DynamicPromptComparison(
        api_key=api_key,
        model="llama3-8b-8192",  # Consistent model choice
        rate_limit_delay=0.5  # Adjust as needed for rate limits
    )

    # --- CHOOSE YOUR BENCHMARK ---
    # To run HLE: benchmark_name = "HLE"
    # To run AIME: benchmark_name = "AIME"
    benchmark_name = "AIME" # Default to AIME, user can change this

    test_inputs = []
    output_filename = "results.json" # Default

    if benchmark_name == "HLE":
        # Assumes HLE data is in a subdirectory 'hle/data/' relative to this script
        # User needs to clone it: git clone https://github.com/centerforaisafety/hle.git
        hle_data_path = "hle/data/hle_test_set.jsonl" # Path as suggested in issue
        test_inputs = load_hle_questions(hle_data_path, max_questions=args.max_questions)
        output_filename = "hle_results.json"
    elif benchmark_name == "AIME":
        test_inputs = load_aime_questions(max_questions=args.max_questions)
        output_filename = "aime_results.json"
    else:
        print(f"Error: Invalid benchmark name '{benchmark_name}'. Choose 'HLE' or 'AIME'.")
        return

    if not test_inputs:
        print(f"No questions loaded for benchmark {benchmark_name}. Exiting.")
        return

    num_questions_to_process = len(test_inputs)
    print(f"Starting benchmark: {benchmark_name} with {num_questions_to_process} questions.")

    results = await comparison.compare_batch(test_inputs)

    # Save results
    comparison.save_results(results, output_filename)
    print(f"Benchmark {benchmark_name} finished. Results saved to {output_filename}")

if __name__ == "__main__":
    # Ensure necessary libraries are installed
    try:
        import groq
        import datasets
    except ImportError as e:
        print(f"ImportError: {e}. Please install the required libraries. You might need to run: pip install groq datasets")
    else:
        asyncio.run(main())
