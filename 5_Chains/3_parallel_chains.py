from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv
load_dotenv()

# --- Fix 1: Use correct model names ---
model1 = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
model2 = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.7)

# --- Fix 2: Correct Pydantic Field definitions ---
class NotesOut(BaseModel):
    title: str = Field(description="Title of the Notes.")
    content: str = Field(description="Notes about the topic.")

class QuizOut(BaseModel):
    quiz: str = Field(description="Quiz on the notes.")

class Output(BaseModel):
    title: str = Field(description="Title of the Notes.")
    content: str = Field(description="Notes about the topic.")
    quiz: str = Field(description="Quiz on the notes.")

# parsers
parser1 = PydanticOutputParser(pydantic_object=NotesOut)
parser2 = PydanticOutputParser(pydantic_object=QuizOut)
parser3 = PydanticOutputParser(pydantic_object=Output)

# --- Fix 3: Add format_instructions to templates ---
template1 = PromptTemplate(
    template="Generate me short and simple notes on {text}\n\n{format_instructions}",
    input_variables=['text'],
    partial_variables={'format_instructions': parser1.get_format_instructions()}
)

template2 = PromptTemplate(
    template="Generate quiz on the text {text}\n\n{format_instructions}",
    input_variables=["text"],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

template3 = PromptTemplate(
    template="""Merge the provided notes and quiz into a single document.

Notes:
{notes}

Quiz:
{quiz}

{format_instructions}""",
    input_variables=["notes", "quiz"],
    partial_variables={'format_instructions': parser3.get_format_instructions()}
)

# parallel chain
parallel_chains = RunnableParallel({
    "notes": template1 | model1 | parser1,
    "quiz": template2 | model2 | parser2,
})


merge_chains = template3 | model2 | parser3


chains = parallel_chains | merge_chains

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chains.invoke({"text": text})

print(result)

# Graph visualization
chains.get_graph().print_ascii()