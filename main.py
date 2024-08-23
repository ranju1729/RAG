import os, sys
import RAG.RAG_core as rc

if __name__ == '__main__':

    physics_file = os.path.join(os.getcwd(), 'RAG', 'Physics', 'physics.txt')
    embeddings_dir = os.path.join(os.getcwd(), 'RAG', 'embeddings')

    with open(physics_file, encoding='utf-8-sig') as f:
        text = f.read()

    paragraphs = rc.split_text_into_paragraphs(text, sentences_per_paragraph=5)
    embedded_text = rc.get_embeddings(os.path.join(embeddings_dir,'physics.json'), 'mistral', paragraphs)

    more_prompts = 'Y'
    while more_prompts == 'Y':
        rc.query_rag(embedded_text, paragraphs)
        more_prompts = (input("want to continue (Y/N): "))

    sys.exit(0)