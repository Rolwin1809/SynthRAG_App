import arxiv
import wikipedia
import fitz  
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Initialize once
embedder = SentenceTransformer('all-MiniLM-L6-v2')
generator = pipeline("text-generation", model="gpt2")

def get_wikipedia_content(topic):
    try:
        return wikipedia.page(topic).content
    except Exception as e:
        return f"Error fetching Wikipedia content: {str(e)}"

def get_arxiv_content(query):
    try:
        search = arxiv.Search(query=query, max_results=3)
        summaries = [result.summary for result in search.results()]
        return "\n\n".join(summaries)
    except Exception as e:
        return f"Error fetching ArXiv content: {str(e)}"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def create_faiss_index(text):
    sentences = text.split(". ")
    vectors = embedder.encode(sentences)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, sentences

def generate_response(query, context):
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=150, do_sample=True)[0]["generated_text"]
    return result.split("Answer:")[-1].strip()
