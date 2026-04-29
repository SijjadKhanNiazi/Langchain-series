from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
load_dotenv()
llm1 = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
)

model = ChatHuggingFace(llm=llm1)
parser = StrOutputParser()

template1 = PromptTemplate(
    template = "write a Joke on - {topic}.",
    input_variables=["topic"]   
)
template2 = PromptTemplate(
    template = "Explain this joke : {topic}.",
    input_variables=["topic"]
)

chain = RunnableSequence(template1, model, parser, template2, model, parser)
print(chain.invoke({"topic": "Python programming"}))

 