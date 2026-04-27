from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
    temperature=0.7,
)
Model = ChatHuggingFace(llm=llm)

class person (BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City of the person")

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template= "Write a name, age, and city for a {national} person \n {format_instructions}.",
    input_variables=["national"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
chain = template | Model | parser
final_result = chain.invoke({"national": "Pakistani"})
print(final_result)
