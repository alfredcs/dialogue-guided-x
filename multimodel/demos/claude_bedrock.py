import os
import sys
import boto3
import logging
import io
import json
import base64
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
#from langchain.embeddings import BedrockEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from PIL import Image
from botocore.exceptions import ClientError
# Langchian agent
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.chat_models import BedrockChat
from langchain.agents import initialize_agent, AgentType
#from langchain import FewShotPromptTemplate

## Rewrite
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock

boto3_bedrock = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)
logger = logging.getLogger(__name__)

region = "us-east-1"

def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client

def image_to_base64(img) -> str:
    """Converts a PIL Image or local image file path to a base64 string"""
    if isinstance(img, str):
        if os.path.isfile(img):
            print(f"Reading image from file: {img}")
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise FileNotFoundError(f"File {img} does not exist")
    elif isinstance(img, Image.Image):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Expected str (filename) or PIL Image. Got {type(img)}")

def bedrock_imageGen(model_id:str, prompt:str, iheight:int, iwidth:int, src_image, image_quality:str, image_n:int, cfg:float, seed:int):
    negative_prompts = [
                "poorly rendered",
                "poor background details",
                "poorly drawn objects",
                "poorly focused objects",
                "disfigured object features",
                "cartoon",
                "animation"
            ]
    titan_negative_prompts = ','.join(negative_prompts)
    try:
        if model_id == "amazon.titan-image-generator-v1":
            if cfg > 10.0:
               cfg = 10.0
            if src_image:
                src_img_b64 = image_to_base64(src_image)
                body = json.dumps(
                    {
                        "taskType": "IMAGE_VARIATION",
                        "imageVariationParams": {
                            "text":prompt,   # Required
                            "negativeText": titan_negative_prompts,  # Optional
                            "images": [src_img_b64]
                        },
                        "imageGenerationConfig": {
                            "numberOfImages": image_n,   # Range: 1 to 5 
                            "quality": image_quality,  # Options: standard or premium
                            "height": iheight,         # Supported height list in the docs 
                            "width": iwidth,         # Supported width list in the docs
                            "cfgScale": cfg,       # Range: 1.0 (exclusive) to 10.0
                            "seed": seed             # Range: 0 to 214783647
                        }
                    }
                )
            else:
                body = json.dumps(
                    {
                        "taskType": "TEXT_IMAGE",
                        "textToImageParams": {
                            "text":prompt,   # Required
                            "negativeText": titan_negative_prompts  # Optional
                        },
                        "imageGenerationConfig": {
                            "numberOfImages": image_n,   # Range: 1 to 5 
                            "quality": image_quality,  # Options: standard or premium
                            "height": iheight,         # Supported height list in the docs 
                            "width": iwidth,         # Supported width list in the docs
                            "cfgScale": cfg,       # Range: 1.0 (exclusive) to 10.0
                            "seed": seed             # Range: 0 to 214783647
                        }
                    }
                )
        elif model_id == "stability.stable-diffusion-xl-v1:0":
            style_preset = "photographic"  # (e.g. photographic, digital-art, cinematic, ...)
            clip_guidance_preset = "FAST_GREEN" # (e.g. FAST_BLUE FAST_GREEN NONE SIMPLE SLOW SLOWER SLOWEST)
            sampler = "K_DPMPP_2S_ANCESTRAL" # (e.g. DDIM, DDPM, K_DPMPP_SDE, K_DPMPP_2M, K_DPMPP_2S_ANCESTRAL, K_DPM_2, K_DPM_2_ANCESTRAL, K_EULER, K_EULER_ANCESTRAL, K_HEUN, K_LMS)
            body = json.dumps({
                "text_prompts": (
                        [{"text": prompt, "weight": 1.0}]
                        + [{"text": negprompt, "weight": -1.0} for negprompt in negative_prompts]
                    ),
                    "cfg_scale": cfg,
                    "seed": seed,
                    "steps": 60,
                    "style_preset": style_preset,
                    "clip_guidance_preset": clip_guidance_preset,
                    "sampler": sampler,
                    "width": iwidth,
                })
            
        response = get_bedrock_client(region).invoke_model(
            body=body, 
            modelId=model_id,
            accept="application/json", 
            contentType="application/json"
        )
        response_body = json.loads(response["body"].read())
        if model_id == "amazon.titan-image-generator-v1":
            base64_image_data = response_body["images"][0]
        elif model_id == "stability.stable-diffusion-xl-v1:0":
            base64_image_data = response_body["artifacts"][0].get("base64")

        return base64_image_data

    except ClientError:
        logger.error("Couldn't invoke Titan Image Generator Model")
        raise
            
def bedrock_textGen(model_id, prompt, max_tokens, temperature, top_p, top_k, stop_sequences):
    stop_sequence = [stop_sequences]
    inference_modifier = {
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": stop_sequence,
    }

    textgen_llm = Bedrock(
        model_id=model_id,
        client=boto3_bedrock,
        model_kwargs=inference_modifier,
    )
    
    return textgen_llm(prompt)

def bedrock_textGen_agent(model_id, prompt, max_tokens, temperature, top_p, top_k, stop_sequences):
    stop_sequence = [stop_sequences]
    inference_modifier = {
        "max_tokens_to_sample": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "stop_sequences": stop_sequence,
    }

    textgen_llm = Bedrock(
        model_id=model_id,
        client=boto3_bedrock,
        model_kwargs=inference_modifier,
    )

    ## Using Dickduckgo as search engine
    wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", safesearch='Moderate', time=None, max_results=3)
    duckduckgo_search = DuckDuckGoSearchRun()
    duckduckgo_tool = DuckDuckGoSearchResults()

    # initialize the agent
    agent_chain = initialize_agent(
        [duckduckgo_tool],
        textgen_llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # run the agent
    output = agent_chain.run(
        prompt,
    )
    
    return output

def create_vector_db_chroma_index(bedrock_clinet, chroma_db_path: str, pdf_file_names: str, bedrock_embedding_model_id:str):
    #replace the document path here for pdf ingestion
    loader = PyPDFLoader(pdf_file_name)
    doc = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=200, separator="\n")
    chunks = text_splitter.split_documents(doc)
    emb_model = "sentence-transformers/all-MiniLM-L6-v2"
    '''
    embeddings = HuggingFaceEmbeddings(
        model_name=emb_model,
        cache_folder="./cache/"
    )
    '''
    embeddings = create_langchain_vector_embedding_using_bedrock(bedrock_client=bedrock_client, bedrock_embedding_model_id=bedrock_embedding_model_id)
    db = Chroma.from_documents(chunks,
                               embedding=embeddings,
                               persist_directory=chroma_db_path)
    db.persist()
    return db

def create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=bedrock_embedding_model_id)
    return bedrock_embeddings_client


def create_opensearch_vector_search_client(index_name, opensearch_password, bedrock_embeddings_client, opensearch_endpoint, _is_aoss=False):
    docsearch = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=bedrock_embeddings_client,
        opensearch_url=f"https://{opensearch_endpoint}",
        http_auth=(index_name, opensearch_password),
        is_aoss=_is_aoss
    )
    return docsearch

def create_bedrock_llm(bedrock_client, model_version_id, temperature):
    bedrock_llm = Bedrock(
        model_id=model_version_id,
        client=bedrock_client,
        model_kwargs={'temperature': temperature}
        )
    return bedrock_llm

def bedrock_chroma_rag(llm_model_id, embed_model_id, temperature, max_token):
    bedrock_client = get_bedrock_client(region)
    bedrock_embedding_model_id = embed_model_id
    chroma_db= create_vector_db_chroma_index(bedrock_clinet, os.path.join("./","chroma_rag_db", pdf_file_names, bedrock_embedding_model_id))
    retriever = chroma_db.as_retriever()
    llm = Bedrock(model_id=llm_model_id, client=bedrock_client, model_kwargs={"max_tokens_to_sample": max_toekn, "temperature": temperature})

    template = """\n\nHuman:Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    {context}
    Question: {question}
    \n\nAssistant:"""
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)
    conv_qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False)
    returnval = conv_qa_chain("is application development covered?")
    print(returnval["answer"])



def bedrock_llm(model_id, max_tokens, temperature):
    bedrock_embedding_model_id = 'amazon.titan-embed-text-v1'
    index_name = ''

    bedrock_client = get_bedrock_client(region)
    bedrock_llm = create_bedrock_llm(bedrock_client, model_id)
    bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(bedrock_client, bedrock_embedding_model_id)
    opensearch_endpoint = opensearch.get_opensearch_endpoint(index_name, region)
    opensearch_password = secret.get_secret(index_name, region)
    opensearch_vector_search_client = create_opensearch_vector_search_client(index_name, opensearch_password, bedrock_embeddings_client, opensearch_endpoint)
    



    llm = Bedrock(model_id=model_id, client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':max_tokens, 'temperature': temperature})
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=boto3_bedrock)
    prompt_template = """

    Human: Use the following pieces of context to provide a concise answer to the question at the end. Please think before answering and provide answers only when you find supported evidence. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answewr['result']


def prompt_rewrite(distracted_query, temperature, top_p, max_len):
    bedrock_client = get_bedrock_client(region)
    inference_modifier = {
        #"max_tokens_to_sample": 512,
        "max_gen_len": max_len,
        "temperature": temperature,
        #"top_k": 0.85,
        "top_p": top_p,
    }
    
    bedrock_model = Bedrock(
        model_id="meta.llama2-70b-chat-v1",
        client=bedrock_client,
        model_kwargs=inference_modifier,
    )

    chat_llm = BedrockChat(
            model_id="meta.llama2-70b-chat-v1", client=boto3_bedrock, model_kwargs={"max_tokens_to_sample":max_len, "temperature":temperature, "top_p": top_p}
        )

    # create our examples
    examples = [
        {
            "input": "A cat sitting on a windowsill.",
            "output": "Imagine a cozy, sunlit room, with sheer curtains gently swaying in the breeze. On the wooden windowsill, there's a fluffy, ginger tabby cat lounging lazily. The cat's green eyes are half-closed, basking in the warm sunlight filtering through the window, casting a soft glow on its fur. Potted plants are placed around the windowsill, adding a touch of greenery to the serene scene."
        }, {
            "input": "A futuristic cityscape at night.",
            "output": "Envision a sprawling futuristic cityscape under the cloak of night, illuminated by the neon glow of skyscrapers. Hover cars zip through the skyways, leaving trails of light in their wake. The architecture is a blend of high-tech structures and eco-friendly green buildings with vertical gardens. In the sky, a giant hologram advertisement plays, reflecting off the glossy surface of a nearby tower, while the moon looms large in the starry sky."
        }, {
            "input": "A medieval knight on a quest.",
            "output": "Picture a valiant medieval knight, clad in shining armor, embarking on a noble quest through an ancient forest. The knight rides a majestic, well-armored steed. The dense forest is shrouded in mist, with rays of sunlight piercing through the canopy, creating a mystical ambiance. The knight holds aloft a banner with a crest symbolizing their noble cause, and in the background, an imposing, mysterious castle can be seen atop a distant hill, its flags fluttering in the wind."
        }
    ]
    
    # This is a prompt template used to format each individual example.
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """Your role as an expert prompt engineer involves meticulously refining the input text, transforming it into a detailed and enriched prompt. This refined prompt is destined for a text-to-image generation model. Your primary objective is to maintain the core semantic essence of the original text while infusing it with rich, descriptive elements. Such detailed guidance is crucial for steering the image generation model towards producing images of superior quality, characterized by their vivid and expressive visual nature. Your adeptness in prompt crafting is instrumental in ensuring that the final images not only captivate visually but also resonate deeply with the original textual concept. Here are some examples: 
    """

    # now create the few shot prompt template
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        #prefix=prefix,
        #suffix=suffix,
        #input_variables=["query"],
        #example_separator="\n\n****"
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prefix),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    output_parser = StrOutputParser()
    chain = final_prompt | bedrock_model | output_parser

    return chain.invoke({"input": distracted_query})
    

if __name__ == "__main__":
    response = prompt_rewrite("A man walks his dog toward the camera in a park.", 0.5, 0.85, 512)
    print(response)