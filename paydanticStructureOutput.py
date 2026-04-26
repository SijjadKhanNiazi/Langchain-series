from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from typing import Annotated , Optional
from pydantic import BaseModel, Field 
from dotenv import load_dotenv

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-V4-Pro", 
    task = "text-generation",
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    name: list[str] = Field( description= "name of product")
    details: str = Field(description= "write a short details about the product, if available")
    keyfeatures: list[str] = Field(description= "list of key features not to be more than 5, if available")
    pros: Optional[list[str]] = None
    cons: Optional[list[str]] = None
    cgpa: Optional[float] = Field(None, gt=0.0, lt=4.0) 
structured_model = model.with_structured_output(Review) 
result = structured_model.invoke("""'This style is conversational and descriptive, similar to a review or a casual recommendation. It weaves features into sentences without a strict layout.If you are looking for a top-tier audio experience, the Sony WH-1000XM5 is widely considered one of the best pairs of headphones on the market right now. Sony really focused on silence with these, using two processors and eight microphones to block out everything from plane engines to human voices. They arent just for quiet, though; the new 30mm carbon fiber drivers make your music sound incredibly crisp and detailed. They are super light and covered in soft synthetic leather, so you can wear them all day—which is great because the battery lasts for about 30 hours even with noise canceling turned on. If you are in a rush, a quick 3-minute charge gets you 3 hours of playback. They also have some smart "Speak-to-Chat" features that pause your music the moment you start talking, and they can connect to two devices at once, so you can switch between your laptop and your phone without a headach'""")
    
print(result)