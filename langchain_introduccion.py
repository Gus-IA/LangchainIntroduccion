import langchain
from langchain import HuggingFacePipeline
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import load_tools, initialize_agent, AgentType

# OJO! max_length tiene que ser suficiente como para tener el documento (chuck) + el prompt + el system prompt + respuesta generada !!!
llm = HuggingFacePipeline.from_model_id(
    model_id="OpenAssistant/stablelm-7b-sft-v7-epoch-3",
    task="text-generation",
    model_kwargs={"temperature": 0.0, "max_length": 2048, "device_map": "auto"},
)

template = """<|prompter|>{question}<|endoftext|><|assistant|>"""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is the meaning of life?"

llm_chain.run(question)


loader = OnlinePDFLoader("https://arxiv.org/pdf/1911.01547.pdf")
document = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=64)

# texts = text_splitter.split_text(raw_text)
documents = text_splitter.split_documents(document)

print(len(documents))

documents[10].page_content

documents[11].page_content

embeddings = HuggingFaceEmbeddings()

query_result = embeddings.embed_query(documents[0].page_content)

query_result

vectorstore = Chroma.from_documents(documents, embeddings)

qa = ConversationalRetrievalChain.from_llm(
    llm, vectorstore.as_retriever(), return_source_documents=True
)

chat_history = []
query = "What is the definition of intelligence?"
result = qa({"question": query, "chat_history": chat_history})
result["answer"]

result["source_documents"][0].page_content


tools = load_tools(
    ["arxiv"],
)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


agent_chain.run(
    "What's the paper On the measure of intelligence, by François Chollet, about?",
)
