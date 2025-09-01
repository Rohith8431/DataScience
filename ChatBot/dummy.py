import torch
from transformers import pipeline

pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    device=-1
    )
output = pipeline("translate english to russian: The weather is nice today")
print(output)