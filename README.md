# model-chaining-with-langchain-and-huggingface
This repository demonstrates how to integrate HuggingFace Transformers with LangChain to build a pipeline for text summarization.

## Features
1. **Text Summarization**: Summarizes user-input text into a desired length (`short`, `medium`, or `long`) using HuggingFace's `facebook/bart-large-cnn` model.
2. **Refinement**: Further refines the generated summary using the `facebook/bart-large` model.
3. **Question Answering**: Uses `deepset/roberta-base-squad2` to answer questions based on the summary.

## How It Works
- The script utilizes **LangChain**'s `PromptTemplate` to define a structured summarization prompt.
- The chain is built by connecting the prompt to the summarization pipeline, followed by a refinement step using HuggingFace models.
- After generating the summary, users can ask questions about the summarized content using a question-answering model.

## Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/siddharthlanke/model-chaining-with-langchain-and-huggingface.git
   cd model-chaining-with-langchain-and-huggingface
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Log in to HuggingFace CLI:
   ```bash
   huggingface-cli login
   ```
   Enter your access token. You can generate a read-only access token by:
   - Creating an account or logging in at HuggingFace.
   - Navigating to "Access Tokens" in your account settings and generating a token with read-only permissions.
4. Run the script:
   ```bash
   python3 main.py
   ```
   **Note-** On the first run, the models (facebook/bart-large-cnn, facebook/bart-large, and deepset/roberta-base-squad2) will be downloaded from HuggingFace. Ensure you are logged in to HuggingFace for this step.

## Example Usage
- Input text to summarize.
- Specify the desired summary length (short, medium, or long).
- Ask questions about the summary in a conversational manner.
