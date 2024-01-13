import json
import os
import argparse
import sys
import boto3
import time
# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.load.dump import dumps
from urllib.request import urlretrieve
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.chains import RetrievalQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts import PromptTemplate
import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
#from helper_functions import *

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

boto3_bedrock = bedrock.get_bedrock_client(
    #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)


def auth_opensearch(host,  # serverless collection endpoint, without https://
                    region,
                    profile_name='default',
                    service='aoss'):
    # Get the credentials from the boto3 session
    credentials = boto3.Session(profile_name=profile_name).get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)
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
    llm = Bedrock(
            model_id=model_id, client=boto3_bedrock, model_kwargs={"max_tokens_to_sample":max_token, "temperature":temperature}
    )
    os.makedirs("./temp_data", exist_ok=True)
    for url in files:
        file_path = os.path.join("./temp_data", url.rpartition("/")[2])
        urlretrieve(url, file_path)

    loader = PyPDFDirectoryLoader("./temp_data/")

    documents = loader.load()
    # - in our testing Character split works better with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=chunk_size,
        chunk_overlap=chunk_size*0.1,
    )
    docs = text_splitter.split_documents(documents)
    avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(documents)
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

#---------------

def update_vdb(docs, aoss_host, index_name, profile_name, my_region):
    credentials = boto3.Session(profile_name=profile_name).get_credentials()
    auth = AWSV4SignerAuth(credentials, my_region, 'aoss')
    bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)
    docsearch = OpenSearchVectorSearch.from_documents(
        docs,
        bedrock_embeddings,
        opensearch_url=aoss_host,
        http_auth=auth,
        timeout = 100,
        use_ssl = True,
        bulk_size = 1000, # increased from default 500 to accomodat more pdf files
        verify_certs = True,
        connection_class = RequestsHttpConnection,
        index_name=index_name,
        engine="faiss",
    )
    return docsearch

#-----------------
def do_query(query: str, model_id: str, docsearch, max_token: str, temperature: float):
    llm = Bedrock(
            model_id=model_id, client=boto3_bedrock, model_kwargs={"max_tokens_to_sample":max_token, "temperature":temperature}
    )

    qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={'k': 3}),return_source_documents=True)

    prompt_template = """Human: Use the following pieces of context to provide a concise answer to the question at the end. Please think carefully before answering. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Assistant:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    search_kwargs = {
            #"vector_field": "content-embedding",
            #"text_field": "content",
            "k": 3}
    qa_prompt = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs=search_kwargs),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT, "verbose": False},
        verbose=False
    )
    result = qa_prompt({"query": query})
    return result["result"]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process command line input')
    parser.add_argument('-f', '--pdf_files', type=str, default="https://arxiv.org/pdf/2309.10020.pdf", help='PDF s files in list format')
    parser.add_argument('-k', '--max_token', type=int, default=256, help='Max tokens')
    parser.add_argument('-c', '--chunk_size', type=int, default=4000, help='File split chunk size')
    parser.add_argument('-t', '--temperature', type=float, default=0.5, help='LLM Temperature')
    parser.add_argument('-p', '--profile_name', type=str, default='default', help='AWS credential profile name')
    parser.add_argument('-q', '--query', type=str, default='What is Amazon Q?', help='Query string')
    parser.add_argument('-m', '--model_id', choices=['anthropic.claude-v2:1','meta.llama2-70b-v1:0','amazon.titan-tg1-large'], default='anthropic.claude-v2:1', help='LLM Model')

    args = parser.parse_args()

    chunk_size = args.chunk_size
    temperature = args.temperature
    pdf_files = [args.pdf_files]
    max_token = args.max_token
    model_id = args.model_id
    profile_name=args.profile_name
    query=args.query
    #my_session = boto3.session.Session()
    #my_region = my_session.region_name
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    my_region = os.environ.get("AWS_DEFAULT_REGION", None)
    collection_name = 'bedrock-workshop-rag'
    collection_id = '967j1chec55256z804lj'
    # Check if the aoss/collection exists and if not create one
    '''
    aoss_py_client = auth_opensearch(host = "{}.{}.aoss.amazonaws.com".format(collection_id, my_region),
                            profile_name=profile_name, service = 'aoss', region = my_region)
    if not aoss_py_client.indices.exists(index = collection_name):
        aoss_host = aoss_setup(collection_name)
    else:
        aoss_host = "{}.{}.aoss.amazonaws.com".format(collection_id, my_region)
    '''
    aoss_host = "{}.{}.aoss.amazonaws.com:443".format(collection_id, my_region)
    # Inject data to aoss
    docs, avg_doc_length = data_prep(pdf_files, chunk_size)
    print(f'Avg doc length: {avg_doc_length}')
    docsearch = update_vdb(docs, aoss_host, collection_name, profile_name, my_region)

    # Do RAG with dialogue
    response = do_query(query, model_id, docsearch, max_token=256, temperature=0.5)
    print(response)
