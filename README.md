# Knowledge Graph Generator with llama-3.3-70b-versatile

A Streamlit application that extract graph data (entities and relationships) from text input using LangChain and Groq API  llama-3.3-70b-versatile model, and generates interactive graphs.

![alt text](<Screenshot (39).png>)

**The grpah which is generated with txt documet is also their in the Repo named as THE DATAPIPELINE MANAGER**

![alt text](<Screenshot (40).png>)

✅ Entities & relationships I extract for my knowledge graph:
Classes: ETLProcessor, ModelTrainer, Logger, Notifier
Functions/methods: extract, transform, load, train_model, evaluate_model, send_email
Parameters & return types: source: str, df: pd.DataFrame, mode: str, etc.
Dependencies: pandas, numpy, scikit-learn, xgboost
Workflow relations: calls, orchestrated-by, retries-on-failure, triggers-notification

## Features

- Two input methods: text upload (.txt files) or direct text input
- Interactive knowledge graph visualization
- Customizable graph display with physics-based layout
- Entity relationship extraction powered by OpenAI's gpt-4o model

## Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key
- Model name - llama-3.3-70b-versatile

### Dependencies

The application requires the following Python packages:

# Core LangChain packages
langchain>=0.3.0
langchain-core>=0.3.0
langchain-experimental>=0.0.45
langchain-groq>=0.1.0  # ✅ Use Groq instead of OpenAI

# Environment variable support
python-dotenv>=1.0.0

# Graph visualization
pyvis>=0.3.2

# Web UI
streamlit>=1.32.0

Install all required dependencies using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the Streamlit app:

```bash
streamlit run app.py
```

This will start the application and open it in your default web browser (typically at http://localhost:8501)
or 
streamlit run app.py --server.port 8502 (u can name any other port)

## Usage

1. Choose your input method from the sidebar (Upload txt or Input text)
2. If uploading a file, select a .txt file from your computer
3. If using direct input, type or paste your text into the text area
4. Click the "Generate Knowledge Graph" button
5. Wait for the graph to be generated (this may take a few moments depending on the length of the text)
6. Explore the interactive knowledge graph:
   - Drag nodes to rearrange the graph
   - Hover over nodes and edges to see additional information
   - Zoom in/out using the mouse wheel
   - Filter the graph for specific nodes and edges.

## How It Works

The application uses LangChain's experimental graph transformers with OpenAI's gpt-4o model to:
1. Extract entities from the input text
2. Identify relationships between these entities
3. Generate a graph structure representing this information
4. Visualize the graph using PyVis, a Python interface for the vis.js visualization library
