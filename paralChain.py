from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
load_dotenv()
llm1 = HuggingFaceEndpoint(
    repo_id= "MiniMaxAI/MiniMax-M2.7",
    task = "text-generation",
)
llm2 = HuggingFaceEndpoint(
    repo_id= "deepseek-ai/DeepSeek-R1",
    task = "text-generation",
)
Model1 = ChatHuggingFace(llm=llm1)
Model2 = ChatHuggingFace(llm=llm2)
parser = StrOutputParser()

template1 = PromptTemplate(
    template = "write a notes for on this topic:  {topic}.",
    input_variables=["topic"]   
)
template2 = PromptTemplate(
    template = "write a 5 question from this text: {topic}.",
    input_variables=["topic"]
)
template3 = PromptTemplate(
    template = "Merge these two texts: {notes_chain} and {question_chain}.",
    input_variables=["notes_chain", "question_chain"]
)
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    "notes_chain": template1 | Model1 | parser,
    "question_chain": template2 | Model2 | parser
})
merge_chain = template3 | Model1 | parser
chain = parallel_chain | merge_chain
topic = """Welcome to the complete practice test designed to help you put together your exam on Python Scikit-Learn.Whether you're familiar with terms like scikit-learn, scikit, scikit-learn, or scikitlearn, or are delving into the realm of machine learning with Scikit-Learn for the first time, you've arrived at the right place to enhance your knowledge and readiness. This test is adapted to help you study the basics and advanced concepts of the usage of Python's Scikit-Learn library.



Python Scikit-Learn Outlines

Category: Simple

Basic Concepts

Fundamental Algorithms

Category: Intermediate

Model Evaluation and Validation

Feature Engineering and Selection

Category: Complex

Advanced Algorithms

Specialized Topics



Importance of Python Scikit-Learn in Machine Learning

In the landscape of machine learning, the significance of Python's Scikit-Learn library is paramount. It serves as a cornerstone for both beginners and seasoned practitioners, offering a rich array of tools and algorithms to facilitate various facets of machine learning endeavors. Scikit-Learn provides an accessible entry point for newcomers due to its user-friendly interface and comprehensive documentation.

For experienced professionals, its versatility and efficiency in implementing diverse machine learning models make it an indispensable asset. Mastering Scikit-Learn not only empowers one to effortlessly preprocess data, select features, and build models but also enables a deeper understanding of machine learning concepts.

Its extensive capabilities in classification, regression, clustering, and more underscore its pivotal role in shaping the landscape of contemporary machine learning practices. The proficiency gained through exploring and leveraging Scikit-Learn empowers individuals to navigate and excel in the dynamic realm of machine learning applications."""

final_result = chain.invoke({'topic': topic})
print(final_result)
chain.get_graph().print_ascii()
