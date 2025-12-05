from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

# model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# output parser
parser = StrOutputParser()

class Feedback(BaseModel):

      sentiment : Literal['positie', 'negative'] = Field(description = "give the feedback with the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object = Feedback)

# prompt template
template = PromptTemplate(
    template="classify the sentiment of the following feedback text into positivr or negative \n {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables = {"format_instructions": parser2.get_format_instructions}
)

template2 = PromptTemplate(
      template = "Write an apprpriate simple responce for this positive feedback \n {feedback}",
      input_variables = ["feedback"]
)

template3 = PromptTemplate(
      template = "Write an apprpriate simple responce for this nagative feedback \n {feedback}",
      input_variables = ["feedback"]
)


feedback = """
This is an terrible phone
"""

# classifier chain

classifier_chain = template | model | parser2

# branch chain
branch_chain = RunnableBranch( 
      (lambda x : x.sentiment == 'positive', template2 | model | parser),
      (lambda x : x.sentiment == 'negative', template3 | model | parser),
      RunnableLambda(lambda x : "could not find the sentiment")
)

# main chain
chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": feedback})

print(result)
chain.get_graph().print_ascii()