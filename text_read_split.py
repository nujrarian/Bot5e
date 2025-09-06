from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

loader = PyPDFLoader("SRD-OGL_V5.1.pdf")
docs = loader.load()

for doc in docs:
    doc.page_content = doc.page_content.replace('\xa0', '')  # Replace non-breaking space with a regular space
    doc.page_content = doc.page_content.replace('\t', '')  # Replace tabs with a space
    doc.page_content = doc.page_content.replace('\r', '')  # Remove carriage returns


text_splitter = CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings()

llm = OllamaLLM(model="llama3.1")

db = FAISS.from_documents(texts, embeddings)

prompt = ChatPromptTemplate.from_messages([
  ("system", "Answer any use questions based solely on the context below:<context>{context}</context>"),
  ("placeholder", "{chat_history}"),
  ("human", "{input}"),
])

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(db.as_retriever(), combine_docs_chain)
print("Chain Created")
result = rag_chain.invoke({"input": "How does divine smite work?"})
print(result['answer'])


