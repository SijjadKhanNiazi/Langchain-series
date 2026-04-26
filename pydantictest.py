from pydantic import BaseModel, EmailStr, Field
from typing import Optional
class Review(BaseModel):
    name: str = "sijjad"
    details: str
    keyfeatures: list[str]
    email: EmailStr
    pros: Optional[list[str]] = None
    cons: Optional[list[str]] = None
    cgpa: Optional[float] = Field(None, gt=0.0, lt=4.0)

data = {
    "name": "Sony WH-1000XM5",
    "details": "These headphones are meticulously designed to provide industry-leading noise cancellation.",
    "keyfeatures": ["Industry-leading noise cancellation", "Crisp sound quality", "Long battery life"],
    "cgpa": 3.74,
    "email": "sijjad@example.com"
}
    
review = Review(**data)
review_dict = dict(review)  
review_json = review.model_dump_json()

print(review_json)
