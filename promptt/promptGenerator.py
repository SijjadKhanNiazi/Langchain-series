from langchain_core.prompts import PromptTemplate

# 1. Define the template correctly
template = PromptTemplate(
    input_variables=["user_question", "tone", "style", "audience", "language"],
    template="""You are an intelligent assistant. 

Question: {user_question}

Instructions:
- Provide the answer DIRECTLY. 
- Do NOT repeat these instructions or provide a "plan" for the answer.
- Start immediately with the content.
- Answer in a {tone} tone.
- Provide output in {style}.
- Target audience is {audience}.
- Respond in {language}.
- Make sure the answer is clear, accurate, and helpful.""",
    validate_template=True
)

# 2. Save the template
template.save("prompt_template.json")

print("Template saved successfully!")