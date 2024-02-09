from langchain.agents import Tool
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain.llms.bedrock import Bedrock
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (RetrievalQA, ConversationalRetrievalChain, ConversationChain)
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import VectorStoreRetrieverMemory
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.agents import AgentType, initialize_agent
from utils import bedrock, print_ww
import sys, os, argparse, re

module_path = ".."
sys.path.append(os.path.abspath(module_path))

boto3_bedrock = bedrock.get_bedrock_client(
    #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)
openai_api_key = openai_api_key = os.getenv("openai_api_token")
embeddings = OpenAIEmbeddings()

class DocumentInput(BaseModel):
    question: str = Field()
        
def mm_rag_interact(model_id, query, pdf_list, web_list, temperature, max_tokens, top_p): 

    if 'gpt' in model_id.lower():
        llm = ChatOpenAI(model=model_id, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    elif 'claude' in model_id.lower():
        llm = BedrockChat(
            model_id=model_id, 
            #streaming=True,
            #callbacks=[StreamingStdOutCallbackHandler()],
            #client=boto3_bedrock,
            model_kwargs={'temperature': 0.1, 'max_tokens_to_sample': 256},
        )
    tools = []
    files = []
    if pdf_list:
        for i, s in enumerate(pdf_list, 1):
            basename = os.path.basename(s).split('.')[0]
            files.append({"name": basename, "PDFpath": s})
    if web_list:
        for i, s in enumerate(web_list, 1):
            new_s = re.sub(r'default\..*$', '', s)
            basename = new_s.rstrip('/').split('/')[-1]
            files.append({"name": basename, "WEBpath": s})
    if not pdf_list and not web_list:
        return None
    pdf_files = [
        {
            "name": "Alphabet-earnings",
            "PDFpath": "https://abc.xyz/assets/95/eb/9cef90184e09bac553796896c633/2023q4-alphabet-earnings-release.pdf",
        },
        {
            "name": "Amazon-earnings",
            "PDFpath": "https://s2.q4cdn.com/299287126/files/doc_financials/2023/q4/AMZN-Q4-2023-Earnings-Release.pdf",
        },
        {
            "name": "Meta-earning",
            "WEBpath": "https://investor.fb.com/investor-news/press-release-details/2024/Meta-Reports-Fourth-Quarter-and-Full-Year-2023-Results-Initiates-Quarterly-Dividend/default.aspx",
        },
        {
            "name": "Tesla-earning",
            "WEBpath": "https://electrek.co/2024/01/24/tesla-tsla-q4-2023-results/",
        },
    ]
    
    for file in files:
        loader = PyPDFLoader(file["PDFpath"]) if 'PDFpath' in file else WebBaseLoader(file["WEBpath"])
        #loader = MergedDataLoader(loaders=[loader_pdf, loader_web])
        pages = loader.load_and_split()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(pages)
        updater = FAISS.from_documents(docs, embeddings)

        # Save to a local
        updater.save_local("local_faiss_index")

        # THen load the local index
        retriever =  FAISS.load_local("local_faiss_index", embeddings).as_retriever()
        # Wrap retrievers in a Tool
        tools.append(
            Tool(
                args_schema=DocumentInput,
                name=file["name"],
                description=f"useful when you want to answer questions about {file['name']}",
                func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
            )
        )
    
    agent = initialize_agent(
        agent=AgentType.OPENAI_FUNCTIONS, #AgentType.OPENAI_MULTI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=True,
    )
    
    response = agent({"input": query})
    return response
    
def update_faiss_store(pdf_list, web_list):
    files = []
    if pdf_list:
        for i, s in enumerate(pdf_list, 1):
            basename = os.path.basename(s).split('.')[0]
            files.append({"name": basename, "PDFpath": s})
    if web_list:
        for i, s in enumerate(web_list, 1):
            new_s = re.sub(r'default\..*$', '', s)
            basename = new_s.rstrip('/').split('/')[-1]
            files.append({"name": basename, "WEBpath": s})
    if not pdf_list and not web_list:
        return None

    for file in files:
        try:
            loader = PyPDFLoader(file["PDFpath"]) if 'PDFpath' in file else WebBaseLoader(file["WEBpath"])
            #loader = MergedDataLoader(loaders=[loader_pdf, loader_web])
            pages = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(pages)
            updater = FAISS.from_documents(docs, embeddings)
    
            # Save to a local
            updater.save_local("local_faiss_index")
        except:
            print(f"Embedding {file} to vectorDB has failed!")
            pass

def do_faiss_query(model_id, query, temperature, max_tokens, top_p): 
    class DocumentInput(BaseModel):
        question: str = Field()
    tools = []
    if 'gpt' in model_id.lower():
        llm = ChatOpenAI(model=model_id, max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    elif 'claude' in model_id.lower():
        llm = BedrockChat(
            model_id=model_id, 
            #streaming=True,
            #callbacks=[StreamingStdOutCallbackHandler()],
            #client=boto3_bedrock,
            model_kwargs={'temperature': temperature, 'max_tokens_to_sample': max_tokens, "top_p": top_p},
        )

    retriever =  FAISS.load_local("local_faiss_index", embeddings).as_retriever(search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents(query)
    # In actual usage, you would set `k` to be a higher value, but we use k=1 to show that
    # the vector lookup still returns the semantically relevant information
    memory = VectorStoreRetrieverMemory(retriever=retriever)

    # Using a chain
    _DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
        
    Relevant pieces of previous conversation:
    {history}
    
    (You do not need to use these pieces of information if not relevant)
    
    Current conversation:
    Human: {input}
    AI:"""
    PROMPT = PromptTemplate(
        input_variables=["history", "input"], template=_DEFAULT_TEMPLATE
    )
    conversation_with_summary = ConversationChain(
        llm=llm,
        prompt=PROMPT,
        memory=memory,
        verbose=True
    )
    response = conversation_with_summary.predict(input=query)

    '''
    # Wrap retrievers in a Tool
    tools.append(
        Tool(
            args_schema=DocumentInput,
            name=query,
            description=f"useful when you want to answer questions about {query}",
            func=RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
        )
    )

    tool = create_retriever_tool(
        RetrievalQA.from_chain_type(llm=llm, retriever=retriever),
        name="Answer question from vector store",
        description="You are a help assistant. Please check evidence before answer",
    )
    tools = [tool]

    # initialize conversational memory
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )

    agent = initialize_agent(
        agent='chat-conversational-react-description', #AgentType.OPENAI_FUNCTIONS, #AgentType.OPENAI_MULTI_FUNCTIONS,
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3,
        early_stopping_method='generate',
        memory=conversational_memory
    )
    
    response = agent({"input": query})
    '''
    return response, docs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process command line input')
    parser.add_argument('-f', '--pdf_files', type=str, help='PDF files seperated by ,')
    parser.add_argument('-w', '--web_links', type=str, help='Web links')
    parser.add_argument('-q', '--query', type=str, help='Query string')
    parser.add_argument('-m', '--model_id', default='gpt-3.5-turbo-0125', type=str, help='Query string')
    args = parser.parse_args()
    pdf_files = args.pdf_files.split(",") if args.pdf_files else None
    web_links = args.web_links.split(",") if args.web_links else None
    if pdf_files or web_links:
        update_faiss_store(pdf_files, web_links)
    #response = mm_rag_interact(args.model_id, args.query, pdf_files, web_links, temperature=0.01, max_tokens=512, top_p=0.90)
    if args.query:
        response, docs = do_faiss_query(args.model_id, args.query, temperature=0.01, max_tokens=512, top_p=0.90)
        print(response, docs)
