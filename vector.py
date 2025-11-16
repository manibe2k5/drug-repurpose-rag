import pdfplumber

def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

pdf_files = ["data/tigecycline-accord-epar-all-authorised-presentations_en.pdf", "data/tigecycline-accord-epar-medicine-overview_en.pdf"]  # List your PDF files
documents = [extract_text(pdf) for pdf in pdf_files]

import nltk
import os

#nltk.download('punkt')
nltk.download('punkt_tab')

#nltk_data_path = "C:\\GEN AI\\drug-repurposing-rag\\.venv"
#os.path.join(os.getcwd(), 'venv', 'nltk_data')
#nltk.data.path.append(nltk_data_path)

def chunk_text(text, max_length=500):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

chunked_docs = [chunk_text(doc) for doc in documents]

from sentence_transformers import SentenceTransformer

# Set proxy environment variables
os.environ['HTTP_PROXY'] = 'http://lev2-proxy.bayerbbs.net:8080'  # Replace with your proxy address
os.environ['HTTPS_PROXY'] = 'http://lev2-proxy.bayerbbs.net:8080'  # Replace with your proxy address

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = [model.encode(chunks) for chunks in chunked_docs]

from pinecone import Pinecone, ServerlessSpec
import requests


pc = Pinecone(api_key="pcsk_3dcc3q_MaAvbwY95ChKqa2S4tu7ErEsiS3RxPcsiBde8oV97dNz3wLJyqyCmgu8FfRYhJi", proxies = proxies)


try:
    print("Testing proxy connectivity to api.pinecone.io...")
    headers = {'api-key': "pcsk_3dcc3q_MaAvbwY95ChKqa2S4tu7ErEsiS3RxPcsiBde8oV97dNz3wLJyqyCmgu8FfRYhJi"}
    response = requests.get('https://api.pinecone.io', headers=headers, proxies=proxies, timeout=10)
    print(f"Proxy test response: {response.status_code}")
except Exception as e:
    print(f"Proxy test failed: {str(e)}")

index_name = "drug-repurposing"

pc.create_index(
    name=index_name,
    dimension=384, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)

print(pc.list_indexes())



def upload_to_pinecone(index, embeddings, chunked_docs, pdf_files):
    vectors = []
    for doc_idx, (doc_embeds, chunks) in enumerate(zip(embeddings, chunked_docs)):
        for chunk_idx, (embed, chunk) in enumerate(zip(doc_embeds, chunks)):
            vector_id = f"{pdf_files[doc_idx]}_chunk_{chunk_idx}"
            vectors.append((vector_id, embed.tolist(), {"text": chunk, "source": pdf_files[doc_idx]}))
    index.upsert(vectors=vectors)

upload_to_pinecone(index, embeddings, chunked_docs, pdf_files)


query = "What diseases might Tigecycline be effective against?"
query_embed = model.encode([query])[0]
results = index.query(query_embed.tolist(), top_k=5, include_metadata=True)
for match in results['matches']:
    print(match['metadata']['text'])