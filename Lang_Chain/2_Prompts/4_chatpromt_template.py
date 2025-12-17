from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

chat_template = ChatPromptTemplate([
      ('system', "You are a helpful {domain} expert."),
      ('human', "Explain {topic} in simple terms with clearly and well formatted way.")
      # SystemMessage(content="You are a helpful {domain} expert."),                                    # These two things are not able to do its work here.
      # HumanMessage(content = "Explain {topic} in simple terms with clearly and well formatted way.")
])

prompt = chat_template.invoke({
      "domain": "Cricket",
      "topic": "Dusra"
})

print(prompt)