from langchain_core import chat_history
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
template = ChatPromptTemplate([
    ("system", "You are a expert customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])
chat_history = []
with open("chathistory.txt") as file:
    chat_history.extend(file.readlines())
prompt = ChatPromptTemplate.invoke(template, {"chat_history": chat_history, "question": "what is the status of my order?"})

print(prompt)
