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
parser = StrOutputParser()
template1 = PromptTemplate(
    template = "write a detail  on {topic}.",
    input_variables=["topic"]   
)
template2 = PromptTemplate(
    template = "write a 5 line summary of {text}.",
    input_variables=["text"]
)
chain = template1 | Model | parser | template2 | Model | parser
final_result = chain.invoke({"topic": "Artificial Intelligence"})
print(final_result)
chain.get_graph().print_ascii()
