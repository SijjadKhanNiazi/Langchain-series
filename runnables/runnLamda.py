from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableParallel
from dotenv import load_dotenv
load_dotenv()
llm1 = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
)
Model1 = ChatHuggingFace(llm=llm1)
parser = StrOutputParser()
Prompt = PromptTemplate(
    template = "write a joke for on this topic:  {topic}.",
    input_variables=["topic"]   
)
joke_generator = RunnableSequence(Prompt, Model1, parser)
parachain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(lambda x: len(x.split()))
})
final_chain = RunnableSequence(joke_generator, parachain)

finalchainfinal = final_chain.invoke({"topic": "indian food"})