# Example configuration file for SunFlowerAI benchmarking
# Copy this file to config.py and update with your settings

# Groq API Configuration
GROQ_API_KEY = ""  # Get from https://console.groq.com/keys
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Available models: llama3-8b-8192, llama3-70b-8192, etc.
GROQ_BASE_URL = None  # Use None for default Groq endpoint

# Hugging Face Configuration  
HF_TOKEN = ""  # Get from https://huggingface.co/settings/tokens
                            # Required for HLE dataset access
                            # Set to None if you don't need HLE access
                            # Example: "hf_1234567890abcdef1234567890abcdef12345678"

# Rate limiting (seconds between API calls)
# Increase if you hit rate limits
RATE_LIMIT_DELAY = 0.5

# API Parameters
API_TEMPERATURE = 0.7  # Controls randomness (0.0 = deterministic, 1.0 = very random)
API_MAX_TOKENS = 1024  # Maximum tokens in response

# Prompt Engineering
DIRECT_ANSWER_INSTRUCTION = "Now answer the input below as directly and cleanly as you can without any added explanations:"

# Benchmark Settings
DEFAULT_BENCHMARK = "HLE"  # Default benchmark ("AIME" or "HLE")

# File Paths
AIME_RESULTS_FILE = "aime_results.json"  # Output file for AIME results
HLE_RESULTS_FILE = "hle_results.json"   # Output file for HLE results

# Dataset Configuration
AIME_DATASET_NAME = "opencompass/AIME2025"  # Hugging Face dataset name
AIME_CONFIGS = ["AIME2025-I", "AIME2025-II"]  # Both parts of AIME 2025
HLE_DATASET_NAME = "cais/hle"  # HLE dataset on Hugging Face
