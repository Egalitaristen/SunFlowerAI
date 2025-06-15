# SunFlowerAI

## Benchmarking with Humanity's Last Exam (HLE) and AIME 2025

This section describes how to use the provided scripts to benchmark an AI agent using the Groq API on the Humanity's Last Exam (HLE) and AIME 2025 datasets.

### 1. Prerequisites

*   **Python 3.8+**
*   **Groq API Key**: You need an API key from Groq. Get yours at [https://console.groq.com/keys](https://console.groq.com/keys).
*   **Git**: Required for cloning the HLE repository.

### 2. Setup Instructions

**a. Clone this Repository (if you haven't already):**

```bash
# If you are reading this, you likely have it. If not:
# git clone <repository_url>
# cd <repository_directory>
```

**b. Install Python Dependencies:**

Install the necessary Python libraries using pip:

```bash
pip install groq datasets
```

**c. Clone Humanity's Last Exam (HLE) Repository:**

The HLE benchmark script (`run_my_agent.py` and `evaluate_hle.py`) directly uses data from the official HLE repository. Clone it into the root directory of this project (or ensure the path in `run_my_agent.py` and `evaluate_hle.py` under `load_hle_questions` and `ground_truth_file` respectively points to your HLE data location).

```bash
git clone https://github.com/centerforaisafety/hle.git
```

This will create a directory named `hle` containing the HLE data.

**d. Configure Groq API Key:**

Open the `run_my_agent.py` script and replace the placeholder API key with your actual Groq API key:

```python
# In run_my_agent.py, find this line:
api_key = "YOUR_GROQ_API_KEY"
# Replace it with your key, for example:
# api_key = "gsk_YourActualApiKeyGoesHere"
```

### 3. Running the Benchmarks

**a. Running the Agent to Generate Responses (`run_my_agent.py`):**

The `run_my_agent.py` script will process the questions from the chosen benchmark, query the Groq API, and save the results (including the input question, direct response, generated system prompt, and prompted response) to a JSON file.

*   **To run for AIME 2025:**

    Make sure the `benchmark_name` variable in `run_my_agent.py` is set to `"AIME"` (this is often the default):

    ```python
    # In run_my_agent.py, ensure this line is set for AIME:
    benchmark_name = "AIME"
    ```

    Then, run the script:

    ```bash
    python run_my_agent.py
    ```

    This will create `aime_results.json`.

*   **To run for Humanity's Last Exam (HLE):**

    Change the `benchmark_name` variable in `run_my_agent.py` to `"HLE"`:

    ```python
    # In run_my_agent.py, change this line for HLE:
    benchmark_name = "HLE"
    ```

    Then, run the script:

    ```bash
    python run_my_agent.py
    ```

    This will create `hle_results.json`. It expects the HLE data to be in the `hle/data/hle_test_set.jsonl` path relative to the script.

**b. Evaluating the Results:**

After generating the results files, use the corresponding evaluation scripts.

*   **To evaluate AIME results:**

    Run the `evaluate_aime.py` script. It will load `aime_results.json` and the AIME ground truth data from Hugging Face.

    ```bash
    python evaluate_aime.py
    ```

    The script will print the accuracy and a summary of results.

*   **To evaluate HLE results:**

    Run the `evaluate_hle.py` script. It will load `hle_results.json` and the HLE ground truth data from the cloned `hle` repository.

    ```bash
    python evaluate_hle.py
    ```

    The script will print accuracies for multiple-choice, free-response, and overall performance.

### Important Notes

*   **API Rate Limits**: The `run_my_agent.py` script has a `rate_limit_delay` parameter in the `DynamicPromptComparison` class. If you encounter rate limiting issues with the Groq API, you might need to increase this delay.
*   **Model Selection**: You can change the model used by the Groq API by modifying the `model` parameter in the `DynamicPromptComparison` class within `run_my_agent.py`.
*   **Error Handling**: The scripts include basic error handling. Check the console output for any warnings or errors during execution.
*   **HLE Data Path**: If you clone the HLE repository to a different location than the project root, you will need to update the `hle_data_path` in `run_my_agent.py` and `ground_truth_file` in `evaluate_hle.py`.
