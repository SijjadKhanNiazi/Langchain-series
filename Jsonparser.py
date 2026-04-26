from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
)
Model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()
template1 = PromptTemplate(
    template = "write a detail  on Artificial Intelligence \n {format_instructions}.",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template1 | Model | parser
final_result = chain.invoke({})
print(type(final_result))