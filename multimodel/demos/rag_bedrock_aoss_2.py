import json
import os
import argparse
import sys
import boto3
import glob
import time
# We will be using the Titan Embeddings Model to generate our Embeddings.
#from langchain.embeddings import BedrockEmbeddings
from langchain_openai import (OpenAIEmbeddings, ChatOpenAI)
from langchain_community.chat_models import (BedrockChat, ChatVertexAI)
from langchain_community.embeddings import BedrockEmbeddings, HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain.llms.bedrock import Bedrock
#from langchain_openai import OpenAI
from langchain.load.dump import dumps
from urllib.request import urlretrieve
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
#from langchain.vectorstores import OpenSearchVectorSearch
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
#from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
#from helper_functions import *
# For embedding with HF TEI
import requests
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
import shutil


module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

boto3_bedrock = bedrock.get_bedrock_client(
    #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
my_region = os.environ.get("AWS_DEFAULT_REGION", None)
credentials = boto3.Session(profile_name='default').get_credentials()
auth = AWSV4SignerAuth(credentials, my_region, 'aoss')



def is_url(string):
  # Check if the string starts with a valid URL scheme.
  valid_schemes = ["http://", "https://", "ftp://", "file://"]
  for scheme in valid_schemes:
    if string.startswith(scheme):
      return True

def check_for_urls(input_list):
  # Iterate over the list and check if any elements are URLs.
  for element in input_list:
    if is_url(element):
      return True

  # If no URLs were found, return False.
  return False

def move_file(src_path, dst_path):
  # Check if the source file exists.
  if not os.path.isfile(src_path):
    raise FileNotFoundError(f"File not found: {src_path}")
  # Check if the destination directory exists.
  if not os.path.isdir(dst_path):
    os.makedirs(dst_path)
  # Move the file.
  shutil.move(src_path, dst_path)

def auth_opensearch(host: str):  # serverless collection endpoint, without https://
    # Get the credentials from the boto3 session
    #credentials = boto3.Session(profile_name=profile_name).get_credentials()
    #auth = AWSV4SignerAuth(credentials, region, service)
    # Create an OpenSearch client and use the request-signer
    os_client = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
        timeout=3000
    )
    return os_client


def data_prep(files: list, chunk_size: int):
    '''
    llm = Bedrock(
            model_id=model_id, client=boto3_bedrock, model_kwargs={"max_tokens_to_sample":max_token, "temperature":temperature}
    )
    '''
    temp_path = "./temp_data/"
    def is_pdf_filename(string):
        return string.endswith(".pdf")
    os.makedirs(temp_path, exist_ok=True)
    check_for_pdf_filenames = lambda input_list: any(is_pdf_filename(element) for element in input_list)
    if check_for_urls(files):
        for url in files:
            file_path = os.path.join(temp_path, url.rpartition("/")[2])
            urlretrieve(url, file_path)
    if check_for_pdf_filenames(files) and not check_for_urls(files):
        for file in files:
            move_file(file, temp_path)

    loader = PyPDFDirectoryLoader(temp_path)
    documents = loader.load()
    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=chunk_size*0.1,
    )
    docs = text_splitter.split_documents(documents)
    avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(documents)
    # Remove processed files
    files = glob.glob(temp_path+'/*')
    for f in files:
        os.remove(f)
    return docs, avg_doc_length(documents)

def aoss_setup(collection_id):
    vector_store_name = collection_id
    index_name = f"{collection_id}-index"
    encryption_policy_name = f"{collection_id}-sp"
    network_policy_name = f"{collection_id}-np"
    access_policy_name = f'{collection_id}-ap'
    identity = boto3.client('sts').get_caller_identity()['Arn']

    aoss_client = boto3.client('opensearchserverless')

    security_policy = aoss_client.create_security_policy(
        name = encryption_policy_name,
        policy = json.dumps(
            {
                'Rules': [{'Resource': ['collection/' + vector_store_name],
                'ResourceType': 'collection'}],
                'AWSOwnedKey': True
            }),
        type = 'encryption'
    )

    network_policy = aoss_client.create_security_policy(
        name = network_policy_name,
        policy = json.dumps(
            [
                {'Rules': [{'Resource': ['collection/' + vector_store_name],
                'ResourceType': 'collection'}],
                'AllowFromPublic': True}
            ]),
        type = 'network'
    )

    collection = aoss_client.create_collection(name=vector_store_name,type='VECTORSEARCH')

    while True:
        status = aoss_client.list_collections(collectionFilters={'name':vector_store_name})['collectionSummaries'][0]['status']
        if status in ('ACTIVE', 'FAILED'): break
        time.sleep(10)

    access_policy = aoss_client.create_access_policy(
        name = access_policy_name,
        policy = json.dumps(
            [
                {
                    'Rules': [
                        {
                            'Resource': ['collection/' + vector_store_name],
                            'Permission': [
                                'aoss:CreateCollectionItems',
                                'aoss:DeleteCollectionItems',
                                'aoss:UpdateCollectionItems',
                                'aoss:DescribeCollectionItems'],
                            'ResourceType': 'collection'
                        },
                        {
                            'Resource': ['index/' + vector_store_name + '/*'],
                            'Permission': [
                                'aoss:CreateIndex',
                                'aoss:DeleteIndex',
                                'aoss:UpdateIndex',
                                'aoss:DescribeIndex',
                                'aoss:ReadDocument',
                                'aoss:WriteDocument'],
                            'ResourceType': 'index'
                        }],
                    'Principal': [identity],
                    'Description': 'Easy data policy'}
            ]),
        type = 'data'
    )

    host = collection['createCollectionDetail']['id'] + '.' + os.environ.get("AWS_DEFAULT_REGION", None) + '.aoss.amazonaws.com:443'
    return host

class HuggingFaceTEIEmbeddings(BaseModel, Embeddings):
    """See <https://huggingface.github.io/text-embeddings-inference/>"""
    base_url: str
    normalize: bool = True
    truncate: bool = False
    query_instruction: str
    """Instruction to use for embedding query."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            self.base_url + "/embeddings",
            json={
                "inputs": texts,
                "normalize": self.normalize,
                "truncate": self.truncate,
            },
        )
        return response.json()

    def embed_query(self, text: str) -> list[float]:
        instructed_query = self.query_instruction + text
        return self.embed_documents([instructed_query])[0]

#---------------

def update_vdb(new_docs, text_embedding, aoss_host, index_name, profile_name, my_region):
    docsearch = OpenSearchVectorSearch.from_documents(
        new_docs,
        text_embedding,
        opensearch_url=aoss_host,
        http_auth=auth,
        timeout = 100,
        use_ssl = True,
        bulk_size = 1500, # increased from default 500 to accomodat more pdf files
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        index_name=index_name,
        engine="faiss",
    )
    '''
    vector = OpenSearchVectorSearch(
        embedding_function = text_embedding,
        index_name = index_name,
        http_auth = auth,
        use_ssl = True,
        verify_certs = True,
        bulk_size = 1000,
        timeout = 100,
        http_compress = True, # enables gzip compression for request bodies
        connection_class = RequestsHttpConnection,
        opensearch_url=aoss_host
    )
    vector.add_documents(
        documents = new_docs,
        vector_field = "osha_vector"
    )
    '''
    return docsearch

#-----------------
def do_query(query: str, model_id: str, text_embedding, aoss_host:str, collection_name:str, profile_name:str, max_token: int, temperature: float, top_p: float, top_k: int, my_region: str):
    '''
    credentials = boto3.Session(profile_name=profile_name).get_credentials()
    auth = AWSV4SignerAuth(credentials, my_region, 'aoss')
    #awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
    print(f'Auth toekns: {auth}')
    '''
    vectordb = OpenSearchVectorSearch(index_name=collection_name, 
                                                 embedding_function=text_embedding, 
                                                 opensearch_url=aoss_host,
                                                 http_auth=auth,
                                                 timeout = 100,
                                                 use_ssl = True,
                                                 verify_certs = True,
                                                 connection_class = RequestsHttpConnection,
                                                 is_aoss=True,
    )
    
    if model_id == 'meta.llama2-70b-v1:0' or model_id == 'meta.llama2-70b-v1':
        llm = Bedrock(
            model_id=model_id, client=boto3_bedrock, model_kwargs={"max_gen_len":max_token, "temperature":temperature, "top_p": top_p}
        )
    elif model_id == 'anthropic.claude-v2' or model_id == 'anthropic.claude-v2:1':
        llm = BedrockChat(
            model_id=model_id, client=boto3_bedrock, model_kwargs={"max_tokens_to_sample":max_token, "temperature":temperature, "top_p": top_p, "top_k": top_k}
        )
        prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. Please think carefully before answering. Your answers should be supported by evidences.
            {context}
            Question: {question}
            Assistant:"""
    elif model_id == 'amazon.titan-text-express-v1':
        llm = Bedrock(
             model_id=model_id, client=boto3_bedrock, model_kwargs={"maxTokenCount":max_token, "temperature":temperature, "top_p": top_p}
        )
    elif model_id == 'gpt-4-1106-preview':
        llm = ChatOpenAI(model_name=model_id,  temperature=temperature, max_tokens=max_token)
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        {context}
        Question: {question}
        Answer: """
    elif model_id == 'gemini-pro':
        llm = ChatVertexAI(model_name=model_id, temperature=temperature, max_output_tokens=max_token, top_p=top_p, top_k=top_k)
        prompt_template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        {context}
        Question: {question}
        Answer: """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_kwargs={'k': 3}),return_source_documents=True)
    search_kwargs = {
            #"vector_field": "content-embedding",
            #"text_field": "content",
            "k": 3}
    qa_prompt = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs=search_kwargs),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT, "verbose": False},
        verbose=False
    )
    result = qa_prompt({"query": query})
    return result["result"]

def main():
    parser = argparse.ArgumentParser(description='Process command line input')
    parser.add_argument('-f', '--pdf_files', nargs="*", type=str, help='PDF s files in list format')
    parser.add_argument('-k', '--max_token', type=int, default=256, help='Max tokens')
    parser.add_argument('-x', '--top_p', type=float, default=0.85, help='Top_P')
    parser.add_argument('-y', '--top_k', type=int, default=40, help='Top_K')
    parser.add_argument('-c', '--chunk_size', type=int, default=500, help='File split chunk size')
    parser.add_argument('-t', '--temperature', type=float, default=0.5, help='LLM Temperature')
    parser.add_argument('-p', '--profile_name', type=str, default='default', help='AWS credential profile name')
    parser.add_argument('-q', '--query', type=str, help='Query string')
    parser.add_argument('-m', '--model_id', choices=['anthropic.claude-v2:1', 'anthropic.claude-v2','meta.llama2-70b-v1', 'meta.llama2-70b-v1:0', 'amazon.titan-text-express-v1', 'gpt-4-1106-preview'], default='anthropic.claude-v2:1', help='LLM Model')

    args = parser.parse_args()

    chunk_size = args.chunk_size
    temperature = args.temperature
    pdf_files = args.pdf_files
    max_token = args.max_token
    top_p = args.top_p
    top_k = args.top_k
    model_id = args.model_id
    profile_name=args.profile_name
    query=args.query

    collection_name = 'bedrock-workshop-rag'
    collection_id = '967j1chec55256z804lj'
    # Check if the aoss/collection exists and if not create one
    '''
    aoss_py_client = auth_opensearch(host = "{}.{}.aoss.amazonaws.com".format(collection_id, my_region))
    if not aoss_py_client.indices.exists(index = collection_name):
        aoss_host = aoss_setup(collection_name)
    else:
        aoss_host = "{}.{}.aoss.amazonaws.com".format(collection_id, my_region)
    '''
    aoss_host = "{}.{}.aoss.amazonaws.com:443".format(collection_id, my_region)

    # Embedding
    #text_embedding = BedrockEmbeddings(client=boto3_bedrock, model_id="amazon.titan-embed-text-v1")
    text_embedding =  OpenAIEmbeddings(openai_api_key=os.getenv('openai_api_token'))
    #text_embedding = HuggingFaceEmbeddings(model_name='intfloat/e5-mistral-7b-instruct', model_kwargs={'device': 'cpu'})
    #text_embedding = HuggingFaceHubEmbeddings(model='http://infs.cavatar.info:8084')
    #text_embedding = HuggingFaceTEIEmbeddings('http://infs.cavatar.info:8084')
    
    # Inject data to aoss
    if pdf_files:
        docs, avg_doc_length = data_prep(pdf_files, chunk_size)
        print(f'PDF file: {pdf_files} and Avg doc length: {avg_doc_length}')
        vectordb = update_vdb(docs, text_embedding, aoss_host, collection_name, profile_name, my_region)

    # Do RAG with dialogue
    if query:
        ##awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)
        ##print(f'Auth toekns: {auth}')
        #vectordb = OpenSearchVectorSearch(index_name=collection_name, 
        #                                         embedding_function=text_embedding, 
        #                                         opensearch_url=aoss_host,
        #                                         http_auth=auth,
        #                                         timeout = 100,
        #                                         use_ssl = True,
        #                                         verify_certs = True,
        #                                         connection_class = RequestsHttpConnection,
        #                                         is_aoss=True,
        #                                 )
        
        response = do_query(query, model_id, text_embedding, aoss_host, collection_name, profile_name, max_token, temperature, top_p, top_k, my_region)
        print(response)
'''
if __name__ == "__main__":
    main()
'''
