from langchain_core.prompts import ChatPromptTemplate
template = ChatPromptTemplate(
   [
        ("system", "You are a expert {role}."),
        ("human", "what is meant by {topic}?")
    ]   

)

prompt = template.invoke({"role": "historian", "topic": "renaissance"})
print(prompt)