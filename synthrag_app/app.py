import streamlit as st
from rag_utils import get_wikipedia_content, get_arxiv_content, extract_text_from_pdf, create_faiss_index, generate_response
import tempfile

st.set_page_config(page_title="SynthRAG Chat", layout="wide")
st.title("🤖 SynthRAG - Your Smart Research Assistant")

# Session state for chat
if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.header("Data Sources")
data_source = st.sidebar.selectbox("Choose Knowledge Source", ["Wikipedia", "arXiv", "Uploaded PDF"])

user_query = st.text_input("Ask a question:", placeholder="e.g., What is quantum entanglement?")

uploaded_pdf = None
if data_source == "Uploaded PDF":
    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if st.button("Get Answer") and user_query:
    with st.spinner("Generating response..."):
        context = ""

        if data_source == "Wikipedia":
            context = get_wikipedia_content(user_query)
        elif data_source == "arXiv":
            context = get_arxiv_content(user_query)
        elif data_source == "Uploaded PDF" and uploaded_pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                tmp_path = tmp_file.name
            pdf_text = extract_text_from_pdf(tmp_path)
            index, sentences = create_faiss_index(pdf_text)
            context = "\n".join(sentences)

        response = generate_response(user_query, context)
        st.session_state.history.append((user_query, response))

# Display chat history
for q, a in reversed(st.session_state.history):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**SynthRAG:** {a}")
