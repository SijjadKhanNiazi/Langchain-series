from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
sentence = ["This is a test sentence for generating embeddings.",
            "Embeddings are useful for various NLP tasks such as clustering, classification, and semantic search."]
embedding_vector = embeddings.embed_documents(sentence)
print(str(embedding_vector))

