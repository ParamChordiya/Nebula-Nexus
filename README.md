# Nebula-Nexus

Nebula-Nexus is a futuristic PDF question-answering assistant for research papers, powered by Retrieval-Augmented Generation (RAG). This tool allows users to upload any PDF and receive precise answers to their questions based on the content of the research paper.

## Features

- Upload a PDF of a research paper and ask questions about its content.
- Leveraging RAG to provide accurate answers directly from the paper.
- Powered by OpenAI's API for advanced natural language understanding.

## Requirements

- Python 3.7 or above
- [Streamlit](https://streamlit.io/)
- [OpenAI API Key](https://platform.openai.com/)

## Installation and Setup

Follow these steps to get Nebula-Nexus running locally.

### 1. Clone the repository

```bash
git clone https://github.com/ParamChordiya/Nebula-Nexus.git
cd Nebula-Nexus
```

### 2. Create a virtual environment

On macOS/Linux:

```bash
python3 -m venv myenv
```

On Windows:

```bash
python -m venv myenv
```

### 3. Activate the virtual environment

On macOS/Linux:
```bash
source myenv/bin/activate
```

On Windows:
```bash
myenv\Scripts\activate
```

### 4. Install the required packages

```bash
pip install -r requirements.txt
```
### 5. Create a .env file
In the root directory, create a .env file and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 6. Run the app
Start the Streamlit app with the following command:
```bash 
streamlit run app.py
```

## Usage
1. Open the provided URL in your browser.
2. Upload a research paper in PDF format.
3. Ask questions, and Nebula-Nexus will retrieve answers based on the paper’s content.