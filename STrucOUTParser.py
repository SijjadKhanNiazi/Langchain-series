from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parser import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
)
Model = ChatHuggingFace(llm=llm)
schemas = [
    ResponseSchema(fact_1= "Fact_1", description="Fact 1 about topic"),
    ResponseSchema(fact_2= "Fact_2", description="Fact 2 about topic"),
    ResponseSchema(fact_3= "Fact_3", description="Fact 3 about topic")
]
parser = StructuredOutputParser.from_response_schemas(schemas)
template1 = PromptTemplate(
    template = "write a detail  on {topic} \n {format_instructions}.",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = template1 | Model | parser
final_result = chain.invoke({"topic": "Artificial Intelligence"})