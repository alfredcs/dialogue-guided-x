# Version: v0.10: adding RAG solution without MM
#

from PIL import Image
import urllib.request
import typing
import os
import sys
import io
import http.client
from operator import itemgetter
import google.generativeai as genai
#from vertexai.preview.generative_models import (GenerativeModel, Part)
from openai import OpenAI
import base64
import requests
from io import BytesIO
import urllib.request
import cv2
from brain import get_index_for_pdf
#from langchain.chat_models import (ChatOpenAI, BedrockChat, ChatVertexAI)
from langchain_community.chat_models import (ChatOpenAI, BedrockChat, ChatVertexAI, ChatHuggingFace)
from langchain_community.llms import HuggingFaceTextGenInference
#from langchain_experimental.chat_models import Llama2Chat
from langchain.chains import LLMChain
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import get_buffer_string  
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import hmac

from rag_bedrock_aoss_lcagent import *
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import initialize_agent, AgentType


module_path = ".."
sys.path.append(os.path.abspath(module_path))
#from claude_bedrock import bedrock_textGen, bedrock_llm, bedrock_imageGen, bedrock_textGen_agent
from utils.gemini_generative_models import _GenerativeModel as GenerativeModel
from utils.gemini_generative_models import Part 


google_api_key = os.getenv("gemini_api_token")
openai_api_key = openai_api_key = os.getenv("openai_api_token")

genai.configure(api_key=google_api_key)
video_file_name = "download_video.mp4"
openai_client = OpenAI(api_key=os.getenv('openai_api_token'))
voice_prompt = ""
chat_history = []
#model = WhisperModel("large-v3")
#vertexai.init(project="proj01-148900", location="us-central1")
temp_audio_file = 'temp_input_audio.mp3'

## VectorDB
profile_name = 'default'
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
my_region = os.environ.get("AWS_DEFAULT_REGION", None)
collection_name = 'bedrock-workshop-rag'
collection_id = '967j1chec55256z804lj'
aoss_host = "{}.{}.aoss.amazonaws.com:443".format(collection_id, my_region)
#credentials = boto3.Session(profile_name='default').get_credentials()
#auth = AWSV4SignerAuth(credentials, my_region, 'aoss')

## Chat memory
memory = ConversationBufferMemory(  
    return_messages=True, output_key="answer", input_key="question"  
)
    

def fetch_image_from_url(url:str):
    with urllib.request.urlopen(url) as url_response:
        # Read the image data from the URL response
        image_data = url_response.read()
        # Convert the image data to a BytesIO object
        image_stream = io.BytesIO(image_data)
        # Open the image using PIL
        return image_stream

def convert_image_to_base64(BytesIO_image):
    # Convert the image to RGB (optional, depends on your requirements)
    rgb_image = BytesIO_image.convert('RGB')
    # Prepare the buffer
    buffered = BytesIO()
    # Save the image to the buffer
    rgb_image.save(buffered, format="JPEG")
    # Get the byte data
    img_data = buffered.getvalue()
    # Encode to base64
    base64_encoded = base64.b64encode(img_data)
    return base64_encoded.decode()


def textGen(model_name, prompt, max_output_tokens, temperature, top_p):
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content

def textGen_agent(model_name, prompt, max_output_tokens, temperature, top_p):
    model = ChatOpenAI(model_name=model_name, 
                       max_tokens=max_output_tokens,
                       temperature=temperature,
                       top_p=top_p,
                      )
    ## Using Dickduckgo as search engine
    wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", safesearch='Moderate', time=None, max_results=3)
    duckduckgo_search = DuckDuckGoSearchRun()
    duckduckgo_tool = DuckDuckGoSearchResults()

    # initialize the agent
    agent_chain = initialize_agent(
        [duckduckgo_tool],
        model,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # run the agent
    output = agent_chain.run(
        prompt,
    )
    return output

def compose_payload(model_name: str, images, prompt: str, max_output_tokens, temperature) -> dict:
    text_content = {
        "type": "text",
        "text": prompt
    }
    image_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{convert_image_to_base64(image)}"
            }
        }
        for image
        in images
    ]
    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [text_content] + image_content
            }
        ],
        "max_tokens": max_output_tokens,
        "temperature": temperature
    }

def getDescription(model_name, prompt, image, max_output_tokens, temperature, top_p):
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {openai_api_key}"
    }

    #payload = compose_payload(model_name=model_name, images=image, prompt=prompt, max_output_tokens=max_output_tokens, temperature=temperature)
    payload = {
      "model": model_name,
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt 
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{convert_image_to_base64(image)}"
              }
            }
          ]
        }
      ],
      "max_tokens": max_output_tokens,
      "temperature": temperature,
      "top_p": top_p,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content'].replace("\n", "",2)


def getDescription2(model_name, prompt, image, image2, max_output_tokens, temperature, top_p):
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {openai_api_key}"
    }

    #payload = compose_payload(model_name=model_name, images=image, prompt=prompt, max_output_tokens=max_output_tokens, temperature=temperature)
    payload = {
      "model": model_name,
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": prompt
            },
            {
              "type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{convert_image_to_base64(image)}"}
            },
            {
              "type": "image_url",
              "image_url": {"url": f"data:image/jpeg;base64,{convert_image_to_base64(image2)}"}
            },
          ],
        },
      ],
      "max_tokens": max_output_tokens,
      "temperature": temperature,
      "top_p": top_p
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content'].replace("\n", "",2)

def videoCaptioning(model_name, prompt, base64Frames, max_tokens, temperature, top_p):
   PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            prompt,
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
        ],
    },
   ]
   params = {
    "model": model_name,
    "messages": PROMPT_MESSAGES,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "top_p": top_p,
   } 
   result = openai_client.chat.completions.create(**params)
   return result.choices[0].message.content

def getBase64Frames(video_file_name):
    video = cv2.VideoCapture(video_file_name)
    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    return base64Frames




# ++++++++++++++ Local deployed models with TGI>=1.4 ++++++++++++++++++++++
    
    
def tgi_imageGen(option, prompt, max_token, temperature, top_p, top_k):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }
    #print(prompt)
    data = {
        "inputs": prompt,
        "parameters": {
            "best_of": 1,
            "decoder_input_details": True,
            "details": True,
            "do_sample": True,
            "max_new_tokens": max_token,
            "repetition_penalty": 1.05,
            "return_full_text": False,
            "seed": 42,
            "stop": ["photographer"],
            "temperature": temperature,
            "top_k": top_k,
            "top_n_tokens": None,
            "top_p": top_p,
            "truncate": None,
            "streaming":True,
            "watermark": True
        }
    }

    response = requests.post(option, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        result = response.json()
        output = result.get('generated_text')
        return output.lstrip()
    else:
        print(f"Error: {response.status_code}")
        return None


def tgi_textGen(option, prompt, max_token, temperature, top_p, top_k):
    llm = HuggingFaceTextGenInference(
        inference_server_url=option,
        max_new_tokens=max_token,
        top_k=top_k,
        top_p=top_p,
        truncate=None,
        #callbacks=callbacks,
        streaming=True,
        watermark=True,
        temperature=temperature,
        repetition_penalty=1.13,
    )

    c_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world class knowledge expert and able to answer any question as a seasoned assistant."),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()
    chain = c_prompt | llm | output_parser
    return chain.invoke({"input": prompt})


def tgi_textGen_memory(option, text_embedding_model, prompt, max_token, temperature, top_p, top_k):
    tgi_llm = HuggingFaceTextGenInference(
        inference_server_url=option,
        max_new_tokens=max_token,
        top_k=top_k,
        top_p=top_p,
        truncate=None,
        #callbacks=callbacks,
        streaming=True,
        watermark=True,
        temperature=temperature,
        repetition_penalty=1.13,
    )
    
    
    #memory = ConversationBufferMemory(  
    #    return_messages=True, output_key="answer", input_key="question"  
    #)
    #load the chat history from the memory
    load_history_from_memory = RunnableLambda(memory.load_memory_variables) | itemgetter(  
        "history"  
    )  
    load_history_from_memory_and_carry_along = RunnablePassthrough.assign(  
        chat_history=load_history_from_memory  
    )
    
    #ask the LLM to enrich the question with context
    rephrase_the_question = (  
        {  
            "question": itemgetter("question"),  
            "chat_history": lambda x: get_buffer_string(x["chat_history"]),  
        }  
        | PromptTemplate.from_template(  
            """You're a personal assistant to the user.  
                Here's your conversation with the user so far:  
                {chat_history}  
                Now the user asked: {question}  
                To answer this question, you need to look up from their notes about """  
        )  
        | tgi_llm  
        | StrOutputParser()  
    )
    
    print(f'rephrased:{rephrase_the_question}')


    # append the final response to the chat history
    final_chain = (  
        load_history_from_memory_and_carry_along  
        | {"standalone_question": rephrase_the_question}  
        | tgi_llm #compose_the_final_answer
    )  
    
    inputs = {"question": prompt}
    output = rephrase_the_question.invoke(inputs) 
    memory.save_context(inputs, {"answer": output.content}) 
    return output  

def dalle3_imageGen(option: str, prompt: str, size: str, quality: str, n_image: int):
    response = openai_client.images.generate(
        model=option,
        prompt=prompt,
        size=size,
        quality=quality,
        n=n_image,
        response_format="url", #or b64_json
    )
    image_url = response.data[0].url
    return image_url