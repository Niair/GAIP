# https://youtu.be/bFB4zqkcatU?si=fT4CxxBg2EcVcDp1 (main)
# https://youtu.be/1h6lfzJ0wZw?si=xTa8McdJ5_xLkEYm

from transformers import pipeline

model = pipeline(task = "text-generation", model="facebook/bart-large-cnn")

response = model("Tell me a short story about a robot learning to love.", max_length=100, do_sample=True, temperature=0.7)

print(response[0]['generated_text'])