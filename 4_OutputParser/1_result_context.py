# you can also use gemini it thismodel does not work

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_core.prompts import PromptTemplate
import torch

model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Load model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

# Set pad token to avoid warnings
tokenizer.pad_token = tokenizer.eos_token

# Create pipeline
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer, 
    max_new_tokens=100,
    temperature=0.7
)

model = HuggingFacePipeline(pipeline=pipe)




# 1st prompt -> detailed response

template1 = PromptTemplate(
      template = "Write an detailed report on {topic}.", # <|im_start|>user\nWrite an detailed report on {topic}.<|im_end|>\n<|im_start|>assistant\n
      input_variables = ['topic']
)


# 2nd prompt -> summary response

template2 = PromptTemplate(
      template = "Write an 5 line summary on the fillowing text. \n {text}",
      input_variables = ['text']
)


prompt1 = template1.invoke({'topic' : 'black hole'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text':result1})
result2 = model.invoke(prompt2)

# print(result1)
# print("\n----------------------------------------------------------------------\n")
print(result2)