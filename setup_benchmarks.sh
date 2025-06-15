#!/bin/bash

# This script automates the setup for running the HLE and AIME benchmarks.

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting benchmark setup..."

# 1. Clone Humanity's Last Exam (HLE) Repository
if [ -d "hle" ]; then
    echo "'hle' directory already exists. Skipping HLE repository cloning."
else
    echo "Cloning Humanity's Last Exam (HLE) repository..."
    git clone https://github.com/centerforaisafety/hle.git
    echo "HLE repository cloned successfully."
fi

# 2. Install Python Dependencies
echo "Installing Python dependencies (groq, datasets)..."
if command -v pip &> /dev/null; then
    pip install groq datasets
    echo "Python dependencies installed successfully."
elif command -v pip3 &> /dev/null; then
    pip3 install groq datasets
    echo "Python dependencies installed successfully using pip3."
else
    echo "Error: pip is not installed. Please install pip and try again." >&2
    exit 1
fi

# 3. Make the script executable (self-awareness!)
# This step is more for the user to know, the file needs to be executable
# if they downloaded it and it lost its permissions.
if [ -f "setup_benchmarks.sh" ]; then
    chmod +x setup_benchmarks.sh
fi

echo ""
echo "Setup complete!"
echo "Next steps:"
echo "1. IMPORTANT: Edit 'run_my_agent.py' to add your Groq API key."
echo "2. Run the agent: python run_my_agent.py --max_questions 5 (for a quick test)"
echo "3. Run an evaluation: python evaluate_aime.py (if you ran AIME)"
