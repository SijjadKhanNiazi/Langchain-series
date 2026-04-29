from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv
load_dotenv()
llm1 = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
)
llm2 = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
)
Model1 = ChatHuggingFace(llm=llm1)
Model2 = ChatHuggingFace(llm=llm2)
parser = StrOutputParser()
template1 = PromptTemplate(
    template = "write a Linkedin post for on this topic:  {topic}.",
    input_variables=["topic"]   
)
template2 = PromptTemplate(
    template = "write a Tweet from this text: {topic}.",
    input_variables=["topic"]
)

parallel_chain = RunnableParallel({
    "Linkedin": RunnableSequence(template1, Model1, parser),
    "Tweet": RunnableSequence(template2, Model2, parser)
})
print(parallel_chain.invoke({"topic": "Python programming"}))