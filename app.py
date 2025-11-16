import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import boto3
import json
import yaml

model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the config.yaml file
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

api_key = config["dev"]["api_key"]

pc = Pinecone(api_key=api_key)

index_name = "drug-repurposing"
index = pc.Index(index_name)

st.title("Drug Repurposing Q&A AI BOT")

#Step 7: Get the question from user 

user_question = st.text_input("Enter your question about Tigecycline:")
#user_question = "What types of bacteria is Tigecycline effective against, particularly in the context of drug-resistant strains?"

#Step 8: Retrive the matching snippets from the vector DB

def query_pinecone(user_query):
    query_embed = model.encode([user_question])[0]

    # Use keyword arguments for the query method
    results = index.query(
        vector=query_embed.tolist(),  # Keyword 'vector' for the query vector
        top_k=5,                     # Keyword 'top_k' for number of results
        include_metadata=True        # Keyword 'include_metadata' to include metadata
    )

    retrieved_texts = [res["metadata"]["text"] for res in results["matches"]]

    return retrieved_texts

#Step 9 : Initialize amazon bedrock , Pass the prompt + the matching snippets + user question 
# as input to the Amazon Titan Text Lite model

#Amazon Titan Text Lite model is a lightweight, cost-effective large language model (LLM) 
# within the Amazon Titan Text family, designed for efficient text summarization


#Step 10: Write the result back to the users


# --- RAG Pipeline Function ---
def rag_pipeline(user_query, retrieved_texts):

    # --- Initialize Bedrock ---
    aws_access_key_id = config["dev"]["aws_access_key_id"]
    aws_secret_access_key = config["dev"]["aws_secret_access_key"]

    bedrock = boto3.client(service_name='bedrock-runtime', region_name='ap-south-1',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key)

    #Format the prompt for the summarization model
    context = "\n".join(retrieved_texts)
    prompt = f"""Based on the following research snippets, answer the user's question, if you don't find matching answer in the following research snippet then answer "No relevant information found".:

    Context:
    {context}

    Question: {user_query}

    Answer:"""

    #Call the Amazon Bedrock Titan Text Express model for summarization
    model_id = 'amazon.titan-text-lite-v1'
    accept = 'application/json'
    content_type = 'application/json'

    body = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,  # Adjust as needed
            "temperature": 0.3,    # Adjust for creativity vs. factualness
            "topP": 0.9             # Adjust for sampling strategy
        }
    }

    try:
        response = bedrock.invoke_model(
            body=json.dumps(body),
            modelId=model_id,
            accept=accept,
            contentType=content_type
        )

        response_body = json.loads(response.get('body').read())
        summary = response_body.get('results')[0].get('outputText')
        st.write(summary)

    except Exception as e:
        print(f"Error calling Bedrock: {e}")
        st.write("No relevant information found.")



if user_question:
    retrieved_texts = query_pinecone(user_question)
    rag_pipeline(user_question, retrieved_texts)