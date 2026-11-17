from langchain_ollama import ChatOllama

llm=ChatOllama(model="llama3.2:latest")

question = input("Enter the question")
response = llm.invoke(question)
print(response.content)

