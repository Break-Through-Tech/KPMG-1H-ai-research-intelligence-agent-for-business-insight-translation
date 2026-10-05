from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")


#check if chunks exist
print(len(chunks))
print(chunks[0])

#vectorstore = Chroma.from_documents(documents = chunks,embedding = embeddings)

#retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})