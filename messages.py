from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",  
    task = "text-generation",
    max_new_tokens=200
)
model = ChatHuggingFace(llm=llm)
messages = [
    SystemMessage(content="You are a helpful assistant."),  
    HumanMessage(content="who was the founder of india?")]
result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print( messages)
