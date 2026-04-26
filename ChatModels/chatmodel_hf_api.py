from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
    max_new_tokens=200
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("who is field marshal of Pakistan now?")

print(result.content)