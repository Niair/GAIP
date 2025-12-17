from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

template = PromptTemplate(
      template = "Write an answer to the following {question} from the following text \n {text}",
      input_variables = ['question', 'text']
)

parser = StrOutputParser()

# we can also pass through more urls buy inserting them in the list and then pass it in the WebBaseLoader

loader = WebBaseLoader("https://www.flipkart.com/apple-macbook-air-m4-16-gb-512-gb-ssd-macos-sequoia-mc7c4hn-a/p/itmdb7ee0ce0e128?pid=COMH9ZWQ389TVPJG&lid=LSTCOMH9ZWQ389TVPJGO1QM12&marketplace=FLIPKART&store=6bo%2Fb5g&srno=b_1_14&otracker=browse&fm=organic&iid=f23e19fe-3e23-4b49-b46b-ccd3d25a3b83.COMH9ZWQ389TVPJG.SEARCH&ppt=None&ppn=None&ssid=ut52toqnk00000001765383156543")

data = loader.load()

# print(len(data))
# print(data[0].page_content)
# print(data[0].metadata)

chain = template | model | parser

result = chain.invoke({
      'question' : 'tell me the best fetures of this mackbook and its price, also try to sell me this in 100 words', 
      'text':data[0].page_content
})

print(result)