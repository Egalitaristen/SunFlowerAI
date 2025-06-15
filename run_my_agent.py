import asyncio
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time
import argparse
from groq import AsyncGroq
from datasets import load_dataset
import config  # Import our configuration file

@dataclass
class ComparisonResult:
    input_text: str
    direct_response: str
    generated_system_prompt: str
    prompted_response: str

class DynamicPromptComparison:
    def __init__(self, api_key: str = None, model: str = None, base_url: str = None,
                 rate_limit_delay: float = None, temperature: float = None, max_tokens: int = None):
        """
        Initialize with Groq API client

        Args:
            api_key: Groq API key (defaults to config.GROQ_API_KEY)
            model: Model name to use (defaults to config.GROQ_MODEL)
            base_url: Base URL for API (defaults to config.GROQ_BASE_URL)
            rate_limit_delay: Delay between API calls in seconds (defaults to config.RATE_LIMIT_DELAY)
            temperature: Temperature for API calls (defaults to config.API_TEMPERATURE)
            max_tokens: Max tokens for API calls (defaults to config.API_MAX_TOKENS)
        """
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        self.base_url = base_url or config.GROQ_BASE_URL
        self.rate_limit_delay = rate_limit_delay if rate_limit_delay is not None else config.RATE_LIMIT_DELAY
        self.temperature = temperature if temperature is not None else config.API_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.API_MAX_TOKENS
        
        # Initialize Groq client
        if self.base_url:
            self.client = AsyncGroq(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = AsyncGroq(api_key=self.api_key)

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
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            # Rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            content = response.choices[0].message.content
            return content if content is not None else "Error: Empty response from API"
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
        """Get response with the generated system prompt and clear instruction to answer directly"""
        # Combine the system prompt with explicit instructions to answer directly
        enhanced_system_prompt = f"""{system_prompt}

{config.DIRECT_ANSWER_INSTRUCTION}"""
        
        return await self._make_api_call(enhanced_system_prompt, user_input)

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

def load_hle_questions(max_questions: int | None = None) -> List[str]:
    """Loads questions from the HLE Hugging Face dataset, with an optional limit."""
    questions = []
    try:
        # Load HLE dataset from Hugging Face with token if provided
        if config.HF_TOKEN and config.HF_TOKEN != "YOUR_HF_TOKEN":
            dataset = load_dataset("cais/hle", split="test", token=config.HF_TOKEN)
        else:
            dataset = load_dataset("cais/hle", split="test")
        print(f"Loading questions from HLE dataset...")
        
        # Handle both IterableDataset and regular Dataset
        if hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__'):
            # IterableDataset - iterate directly
            for i, item in enumerate(dataset):
                if max_questions and i >= max_questions:
                    print(f"Loaded a maximum of {max_questions} questions as requested.")
                    break
                if 'question' in item and isinstance(item['question'], str):
                    questions.append(item['question'])
                else:
                    print(f"Warning: Skipping HLE item due to missing or invalid 'question' field: {item}")
        else:
            # Regular Dataset - use indexing
            try:
                dataset_length = len(dataset)
                max_to_process = min(max_questions or dataset_length, dataset_length)
                
                for i in range(max_to_process):
                    item = dataset[i]
                    if 'question' in item and isinstance(item['question'], str):
                        questions.append(item['question'])
                    else:
                        print(f"Warning: Skipping HLE item due to missing or invalid 'question' field: {item}")
            except TypeError:
                # Fallback if len() doesn't work - treat as iterable
                for i, item in enumerate(dataset):
                    if max_questions and i >= max_questions:
                        print(f"Loaded a maximum of {max_questions} questions as requested.")
                        break
                    if 'question' in item and isinstance(item['question'], str):
                        questions.append(item['question'])
                    else:
                        print(f"Warning: Skipping HLE item due to missing or invalid 'question' field: {item}")

        if max_questions and len(questions) == max_questions:
            pass # Message already printed
        elif max_questions and len(questions) < max_questions:
            print(f"Loaded {len(questions)} questions from HLE dataset (less than max_questions limit of {max_questions}).")
        else:
            print(f"Loaded {len(questions)} questions from HLE dataset.")

    except Exception as e:
        if "gated dataset" in str(e) or "authenticated" in str(e):
            print(f"Error: HLE dataset access denied. The HLE dataset on Hugging Face is gated and requires authentication.")
            print("To access HLE dataset:")
            print("1. Create a Hugging Face account at https://huggingface.co/")
            print("2. Request access to the dataset at https://huggingface.co/datasets/cais/hle")
            print("3. Get your token from https://huggingface.co/settings/tokens")
            print("4. Add your token to config.py: HF_TOKEN = 'your_token_here'")
            print("\nAlternatively, try running AIME benchmark instead:")
            print("python run_my_agent.py --benchmark AIME --max-questions 10")
        elif "Pillow" in str(e) or "PIL" in str(e):
            print(f"Error: Missing required dependency for image processing.")
            print("The HLE dataset contains multimodal questions that require image processing.")
            print("Please install Pillow: pip install pillow")
            print("Then try running the command again.")
        else:
            print(f"Error loading HLE dataset: {e}. Please ensure 'datasets' library is installed and you have internet connectivity.")
        return []
    return questions

def load_aime_questions(max_questions: int | None = None) -> List[str]:
    """Loads questions from the AIME Hugging Face dataset, with an optional limit."""
    questions = []
    try:
        # Load both AIME 2025 I and II datasets
        all_datasets = []
        for config_name in config.AIME_CONFIGS:
            dataset = load_dataset(config.AIME_DATASET_NAME, config_name, split="test")
            all_datasets.append((config_name, dataset))
        
        # Process both datasets
        for dataset_name, dataset in all_datasets:
            print(f"Loading questions from {dataset_name}...")
            
            # Handle both IterableDataset and regular Dataset
            if hasattr(dataset, '__iter__') and not hasattr(dataset, '__len__'):
                # IterableDataset - iterate directly
                for i, item in enumerate(dataset):
                    if max_questions and len(questions) >= max_questions:
                        print(f"Reached maximum of {max_questions} questions as requested.")
                        break
                    if 'question' in item and isinstance(item['question'], str):
                        questions.append(item['question'])
                    else:
                        print(f"Warning: Skipping AIME item due to missing or invalid 'question' field: {item}")
            else:
                # Regular Dataset - use indexing
                try:
                    dataset_length = len(dataset)
                    for i in range(dataset_length):
                        if max_questions and len(questions) >= max_questions:
                            print(f"Reached maximum of {max_questions} questions as requested.")
                            break
                        item = dataset[i]
                        if 'question' in item and isinstance(item['question'], str):
                            questions.append(item['question'])
                        else:
                            print(f"Warning: Skipping AIME item due to missing or invalid 'question' field: {item}")
                except TypeError:
                    # Fallback if len() doesn't work - treat as iterable
                    for i, item in enumerate(dataset):
                        if max_questions and len(questions) >= max_questions:
                            print(f"Reached maximum of {max_questions} questions as requested.")
                            break
                        if 'question' in item and isinstance(item['question'], str):
                            questions.append(item['question'])
                        else:
                            print(f"Warning: Skipping AIME item due to missing or invalid 'question' field: {item}")
            
            # Break if we've reached the max questions limit
            if max_questions and len(questions) >= max_questions:
                break

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
        "--max-questions", "--max_questions",  # Accept both hyphen and underscore versions
        type=int,
        default=None,
        dest="max_questions",  # Store as max_questions regardless of which format is used
        help="Maximum number of questions to process from the benchmark."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=config.DEFAULT_BENCHMARK,
        choices=["AIME", "HLE"],
        help="Which benchmark to run (AIME or HLE)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.GROQ_MODEL,
        help="Model to use for the benchmark."
    )
    args = parser.parse_args()

    # Check if API key is configured
    if config.GROQ_API_KEY == "YOUR_GROQ_API_KEY":
        print("ERROR: Please update GROQ_API_KEY in config.py with your actual Groq API key.")
        print("You can get your API key from: https://console.groq.com/keys")
        return

    comparison = DynamicPromptComparison()

    # Use benchmark from command line argument
    benchmark_name = args.benchmark

    test_inputs = []
    output_filename = "results.json" # Default

    if benchmark_name == "HLE":
        # Load HLE from Hugging Face
        test_inputs = load_hle_questions(max_questions=args.max_questions)
        output_filename = config.HLE_RESULTS_FILE
    elif benchmark_name == "AIME":
        test_inputs = load_aime_questions(max_questions=args.max_questions)
        output_filename = config.AIME_RESULTS_FILE
    else:
        print(f"Error: Invalid benchmark name '{benchmark_name}'. Choose 'HLE' or 'AIME'.")
        return

    if not test_inputs:
        print(f"No questions loaded for benchmark {benchmark_name}. Exiting.")
        return

    num_questions_to_process = len(test_inputs)
    print(f"Starting benchmark: {benchmark_name} with {num_questions_to_process} questions.")
    print(f"Using model: {config.GROQ_MODEL}")

    results = await comparison.compare_batch(test_inputs)

    # Save results
    comparison.save_results(results, output_filename)
    print(f"Benchmark {benchmark_name} finished. Results saved to {output_filename}")

if __name__ == "__main__":
    # Ensure necessary libraries are installed
    try:
        import groq
        import datasets
        import config
    except ImportError as e:
        print(f"ImportError: {e}. Please install the required libraries. You might need to run: pip install groq datasets")
        if "config" in str(e):
            print("Also make sure config.py is in the same directory as this script.")
    else:
        asyncio.run(main())
