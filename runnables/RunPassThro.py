from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()


llm1 = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
    temperature= "0.5",

)
llm2 = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
)
Model1 = ChatHuggingFace(llm=llm1)
Model2 = ChatHuggingFace(llm=llm2)
parser = StrOutputParser()
template1 = PromptTemplate(
    template = "write a joke for on this topic:  {topic}.",
    input_variables=["topic"]   
)

template2 = PromptTemplate(
    template = "write an explaination for this text: {joke}.",
    input_variables=["joke"],
    
)
joke_generator = RunnableSequence(template1, Model1, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explaination": RunnableSequence(template2, Model2, parser)
})
final_chain = RunnableSequence(joke_generator, parallel_chain)
finalchainfinal = final_chain.invoke({"topic": "indian food"})

print(finalchainfinal)