from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = JSONLoader(
    file_path="banks.json",
    jq_schema=".[]",   
    text_content=False
)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print(len(chunks))
print(chunks[0].page_content)