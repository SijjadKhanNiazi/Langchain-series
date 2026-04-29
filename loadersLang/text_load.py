
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()
template = PromptTemplate(
    template = "write a summary of this poem : \n {poem}.",
    input_variables=["poem"]   
)

loader = TextLoader("sijjad.txt", encoding="utf-8")
docs = loader.load()
chain = RunnableSequence(template, model, parser)
final = chain.invoke({"poem": docs[0].page_content})
print(final)
