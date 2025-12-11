from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
# from langchain_community.document_loaders import PyPDFLoader

# text loader
# loader = PyPDFLoader("7_RAG/attention_is_all_you_need.pdf")

# docs = loader.load()

text = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade  # Grade is a float (like 8.5 or 9.2)

    def get_details(self):
        return self.name"

    def is_passing(self):
        return self.grade >= 6.0


# Example usage
student1 = Student("Aarav", 20, 8.2)
print(student1.get_details())

if student1.is_passing():
    print("The student is passing.")
else:
    print("The student is not passing.")

"""

text1 = """
# Project Name: Smart Student Tracker

A simple Python-based project to manage and track student data, including their grades, age, and academic status.


## Features

- Add new students with relevant info
- View student details
- Check if a student is passing
- Easily extendable class-based design


## 🛠 Tech Stack

- Python 3.10+
- No external dependencies


## Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/student-tracker.git

"""

# text splitter
splitter = RecursiveCharacterTextSplitter.from_language(
      language = Language.PYTHON, # JAVA, PHP, JS, HTML, MARKDOWN
      chunk_size = 300,
      chunk_overlap = 0
)

splitter1 = RecursiveCharacterTextSplitter.from_language(
      language = Language.MARKDOWN,
      chunk_size = 200,
      chunk_overlap = 0
)


# chunks = splitter.split_documents(docs)
chunks = splitter.split_text(text)

chunks1 = splitter.split_text(text1)

print(len(chunks))
print(chunks[0])  # the chunk we get is also an document object
print("---------------------------------------------")
print(chunks[1])

print("\n\n ------------------------------------------------------------------------------------------------------ \n\n")

print(len(chunks1))
print(chunks1[0])  # the chunk we get is also an document object
print("---------------------------------------------")
print(chunks1[1])