AI Research Assistant (RAG)

This project is a simple research assistant that answers questions based on your own documents. Instead of relying on external APIs, everything runs locally using embeddings and semantic search.

I built this to better understand how retrieval-augmented generation (RAG) systems work under the hood.

What it does
Loads text files from a local folder
Breaks them into smaller chunks
Converts them into embeddings
Finds the most relevant pieces based on a question
Generates a structured answer using the retrieved context

The output includes:

A short summary
Key insights
Risks or limitations
Supporting evidence from the documents
Why I built this

Most AI tools feel like black boxes. I wanted to build something from scratch that shows how systems like ChatGPT can use external knowledge instead of just relying on pre-trained data.

This project focuses on the core idea behind RAG:

retrieve first, then generate

Tech stack
Python
sentence-transformers
scikit-learn
basic file handling (TXT input)
How to run

Clone the repo and install dependencies:

pip install -r requirements.txt

Then run:

python main.py

You’ll be prompted to enter a question.

Make sure your documents are inside the data/ folder.

Example

Question:

impact of AI in finance

The system retrieves relevant sections from the documents and generates a structured response with supporting context.

Notes
This runs fully locally (no paid API required)
Works best with small to medium document sets
Designed as a learning project, not production-ready
Future improvements
Add PDF parsing
Improve ranking strategy
Add simple UI (Streamlit or web app)
Better evaluation metrics
Final thoughts

This was a hands-on way for me to understand how modern AI systems combine search and generation. It’s a simplified version, but the core ideas are the same as what’s used in real-world applications.
