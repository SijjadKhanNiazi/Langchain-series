from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
)

Model = ChatHuggingFace(llm=llm)
template1 = PromptTemplate(
    template = "write a detail report on {topic}.",
    input_variables=["topic"]
)
template2 = PromptTemplate(
    template = "write a 5 line summary on {heading}.",
    input_variables=["heading"]
)

parser = StrOutputParser()

chain = template1 | Model | parser | template2 | Model | parser

result = chain.invoke({"topic": "Artificial Intelligence"})
print(result)


