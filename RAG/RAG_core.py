import os
from ollama import embeddings, chat
import json
import numpy as np
from numpy.linalg import norm

SYSTEM_PROMPT = """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """


def split_text_into_paragraphs(text, sentences_per_paragraph=25):
    # Split the text into individual sentences
    sentences = text.split('.')

    # Remove any empty strings from the list of sentences
    sentences = [sentence.strip() + '.' for sentence in sentences if sentence.strip()]

    # Group sentences into paragraphs
    paragraphs = []
    for i in range(0, len(sentences), sentences_per_paragraph):
        paragraph = ' '.join(sentences[i:i + sentences_per_paragraph])
        paragraphs.append(paragraph)

    return paragraphs


def get_embeddings(file_name, model_name, chunks):
    embedded_text = load_embeddings(file_name)

    if embedded_text:
        return embedded_text
    else:
        embedded_text = [
            embeddings(model=model_name, prompt=chunk)['embedding'] for chunk in chunks
        ]

        save_embeddings(file_name, embedded_text)
        return embedded_text


def save_embeddings(file_name, embed_data):
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            json.dump(embed_data, f)


def load_embeddings(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            return json.load(f)

    else:
        return False


def embed_promt(prompt):
    return embeddings(model='mistral', prompt=prompt)['embedding']


def search(prompt, data):
    prompt_norm = norm(prompt)
    similarity_scores = [
        np.dot(prompt, item) / (prompt_norm * norm(item)) for item in data
    ]
    return sorted(zip(similarity_scores, range(len(data))), reverse=True)


def query_rag(embedded_text, paragraphs):
    prompt = str(input("Ask your physics question on CBSE X1 part 1 text book"))
    embedded_prompt = embed_promt(prompt)
    similar_data = search(embedded_prompt, embedded_text)[:5]

    response = chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
                           + "\n".join(paragraphs[item[1]] for item in similar_data),
            },
            {"role": "user", "content": prompt},
        ],
    )
    print("\n\n")
    print(response["message"]["content"])
