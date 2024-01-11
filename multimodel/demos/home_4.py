import streamlit as st
from PIL import Image
import urllib.request
import typing
import os
import sys
import io
import http.client
import google.generativeai as genai
#from vertexai.preview.generative_models import (GenerativeModel, Part)
from openai import OpenAI
import base64
import requests
from io import BytesIO
import urllib.request
import cv2
from brain import get_index_for_pdf
from langchain.chat_models import (ChatOpenAI, BedrockChat, ChatVertexAI)
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import hmac
#from streamlit_mic_recorder import mic_recorder, speech_to_text
from audiorecorder import audiorecorder
from faster_whisper import WhisperModel


module_path = ".."
sys.path.append(os.path.abspath(module_path))
from claude_bedrock import bedrock_textGen, bedrock_llm
from utils.gemini_generative_models import _GenerativeModel as GenerativeModel
from utils.gemini_generative_models import Part 

st.set_page_config(page_title="Gemini Pro with Streamlit",page_icon="🩺")
google_api_key = os.getenv("gemini_api_token")
openai_api_key = openai_api_key = os.getenv("openai_api_token")

genai.configure(api_key=google_api_key)
st.title("LVLM Dashboard & Demo")
video_file_name = "download_video.mp4"
client = OpenAI(api_key=os.getenv('openai_api_token'))
voice_prompt = ""
chat_history = []
#model = WhisperModel("large-v3")
#vertexai.init(project="proj01-148900", location="us-central1")
temp_audio_file = 'temp_input_audio.mp3'

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
    response = client.chat.completions.create(
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
   result = client.chat.completions.create(**params)
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

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the passward is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


#@st.cache_data
@st.cache_resource
def create_vectordb(files, filenames):
    # Show a spinner while creating the vectordb
    with st.spinner("Vector database"):
        vectordb = get_index_for_pdf(
            [file.getvalue() for file in files], filenames, openai_api_key
        )
    return vectordb


def get_asr(audio_filename):
    # Set the API endpoint
    url = 'http://infs.cavatar.info:8081/asr?task=transcribe&encode=true&output=txt'
    # Define headers
    headers = {
        'Accept': 'application/json',
        #'Content-Type': 'multipart/form-data'
    }

    # Define the file to be uploaded
    files = {
        'audio_file': (audio_filename, open(audio_filename, 'rb'), 'audio/mpeg')
    }

    # Make the POST request
    response = requests.post(url, headers=headers, files=files)
    return response.text

# ------------------------------

# Check password
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

with st.sidebar:
    st.title(':orange[Multimodal Config] :pencil2:')
    option = st.selectbox('Choose Your Model',('anthropic.claude-v2:1', 'gemini-pro', 'gemini-pro-vision', 'gpt-4-1106-preview', 'gpt-4-vision-preview', 'mistral-8x7b'))

    if 'model' not in st.session_state or st.session_state.model != option:
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.model = option
    
    st.write("Adjust Your Parameter Here:")
    temperature = st.number_input("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    max_token = st.number_input("Maximum Output Token", min_value=0, value =256)
    top_p = st.number_input("Top_p: The cumulative probability cutoff for token selection", min_value=0.1, value=0.85)
    top_k = st.number_input("Top_k: Sample from the k most likely next tokens at each step", min_value=1, value=40)
    #candidate_count = st.number_input("Number of generated responses to return", min_value=1, value=1)
    stop_sequences = st.text_input("The set of character sequences (up to 5) that will stop output generation", value="\n\n\n")
    gen_config = genai.types.GenerationConfig(max_output_tokens=max_token,temperature=temperature, top_p=top_p, top_k=top_k) #, candidate_count=candidate_count, stop_sequences=stop_sequences)

    #st.divider()
    #st.markdown("""<span ><font size=1>Connect With Me</font></span>""",unsafe_allow_html=True)
    #"[Linkedin](https://www.linkedin.com/in/cornellius-yudha-wijaya/)"
    #"[GitHub](https://github.com/cornelliusyudhawijaya)"
    
    st.divider()
    st.header(':green[Image Understanding] :camera:')
    upload_images = st.file_uploader("Upload your Images Here", accept_multiple_files=True, type=['jpg', 'png', 'pdf'])
    image_url = st.text_input("Or Input Image URL", key="image_url", type="default")
    if upload_images:
        #image = Image.open(upload_image)
        for upload_file in upload_images:
            bytes_data = upload_file.read()
            image = Image.open(io.BytesIO(bytes_data))
            st.image(image)
            #image_path = upload_file
            #base64_image = convert_image_to_base64(upload_file)
            #base64_image = base64.b64encode(upload_file.read())
    elif image_url:
        stream = fetch_image_from_url(image_url)
        st.image(stream)
        image = Image.open(stream)

    st.divider()
    st.caption('Image comparisons')
    upload_image2 = st.file_uploader("Upload the 2nd Images for comparison", accept_multiple_files=False, type=['jpg', 'jpeg', 'png'])
    if upload_image2:
        image2 = Image.open(upload_image2)
        #bytes_data2 = upload_file2.read()
        #image22 = Image.open(io.BytesIO(bytes_data))
        st.image(image2)

    st.divider()
    st.header(':green[Video Understanding] :video_camera:')
    upload_video = st.file_uploader("Upload your video Here", accept_multiple_files=False, type=['mp4'])
    video_url = st.text_input("Or Input Video URL with mp4 type", key="video_url", type="default")
    if video_url:
        urllib.request.urlretrieve(video_url, video_file_name)
        video = Part.from_uri(
            uri=video_url,
            mime_type="video/mp4",
        )
        video_file = open(video_file_name, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    elif upload_video:
        video_bytes = upload_video.getvalue()
        with open(video_file_name, 'wb') as f:
            f.write(video_bytes)
        video = Part.from_uri(
            uri=video_file_name,
            mime_type="video/mp4",
        )
        st.video(video_bytes)
    st.divider()
    st.header(':green[Multimodal RAG] :file_folder:')
    upload_docs = st.file_uploader("Upload pdf files", accept_multiple_files=True, type=['pdf', 'xsl', 'doc'])
    if upload_docs:
         upload_doc_names = [file.name for file in upload_docs]
         st.session_state["vectordb"] = create_vectordb(upload_docs, upload_doc_names)
    st.divider()
    st.header(':green[Audio understanding] :microphone:')
    sample_audio = st.file_uploader("Upload a sample audio file for cloning", type=["mp3", "wav"], accept_multiple_files=False)
    if sample_audio:
        sample_audio_bytes = sample_audio.read()
        st.audio(sample_audio_bytes, format="audio/wav")#, start_time=0, *, sample_rate=None)
    #st.caption('Multilingual transcribe')
    #record_audio=mic_recorder(start_prompt="▶️ ", stop_prompt="⏹️" ,key='recorder')

    #st.divider()
    record_audio=audiorecorder(start_prompt="Voice input start:  ▶️ ", stop_prompt="Record stop: ⏹️", pause_prompt="", key=None)
    if len(record_audio)>3:
        record_audio_bytes = record_audio.export().read()
        st.audio(record_audio_bytes, format="audio/wav")#, start_time=0, *, sample_rate=None)
        record_audio.export(temp_audio_file, format="mp3")
    #    segments, info = model.transcribe(temp_audio_file)
    #    for segment in segments:
    #        voice_prompt += segment.text+', '
    #        voice_prompt = voice_prompt[:-2]
    #    os.remove(temp_audio_file)
    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]
        record_audio = None
        #del st.session_state[record_audio]

# ============================================='

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if upload_images or image_url:
    if option == "gemini-pro" or option == "gpt-4-1106" or option == "anthropic.claude-v2:1":
        st.info("Please switch to a vision model")
        st.stop()
    if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            if option == "gemini-pro-vision":
                context = [prompt,image]
                #context = [prompt, image] if upload_images or image_url else ([prompt, video] if video_url else None)
                if upload_image2:
                    context = [prompt,image, image2]
                response=st.session_state.chat.send_message(context,stream=True,generation_config = gen_config)
                response.resolve()
                msg=response.text
            elif option == "gpt-4-vision-preview":
                msg = getDescription(option, prompt, image, max_token, temperature, top_p)
                if upload_image2:
                     msg = getDescription2(option, prompt, image, image2, max_token, temperature, top_p)
            st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
            st.session_state.messages.append({"role": "assistant", "content": msg})
            
            st.image(image,width=350)
            if upload_image2:
                st.image(image2,width=350)
            st.chat_message("assistant", avatar="🌆").write(msg)
elif upload_video or video_url:
    if option == "gemini-pro" or option == "gpt-4-1106" or option == "anthropic.claude-v2:1":
        st.info("Please switch to a vision model")
        st.stop()
    if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            if option == "gemini-pro-vision":
                multimodal_model = GenerativeModel(option)
                context = [prompt, video]
                responses = multimodal_model.generate_content(context, stream=True)
                #response=st.session_state.chat.send_message(context,stream=True,generation_config = gen_config)
                #response.resolve()
                for response in responses:
                    msg += response.text
            elif option == "gpt-4-vision-preview":
                msg = videoCaptioning(option, prompt, getBase64Frames(video_file_name), max_token, temperature, top_p)
            st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
            st.session_state.messages.append({"role": "assistant", "content": msg})

            #video_file = open(video_file_name, 'rb')
            #video_bytes = video_file.read()
            st.video(video_bytes, start_time=0)
            st.chat_message("assistant", avatar="📹").write(msg)
elif upload_docs:
    if option == "gemini-pro-vision" or option == "gpt-4-vision-preview":
        st.info("Please switch to a text model")
        st.stop()
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        vectordb = st.session_state.get("vectordb", None)
        if not vectordb:
            with st.message("assistant"):
                st.write("You need to provide a PDF")
                st.stop()

        # Search the vectordb for similar content to the user's question
        #search_results = vectordb.similarity_search(prompt, k=3)
        #msg = "/n ".join([result.page_content for result in search_results])

        if option == 'gpt-4-1106-preview':
            llm = ChatOpenAI(model_name=option, temperature=temperature)
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)
            qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(), memory=memory)
        elif option == 'gemini-pro':
            llm = ChatVertexAI(model_name=option, temperature=temperature)
            qa_chain = RetrievalQA.from_chain_type(llm,etriever=vectordb.as_retriever())
        elif option == 'anthropic.claude-v2:1':
            llm = Bedrock(model_id=option, client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':200, 'temperature': temperature})

        #qa_chain = ConversationalRetrievalChain.from_llm(llm = llm, 
        #                                    chain_type = "stuff", 
        #                                    condense_question_llm = llm,
        #                                    retriever = vectordb.as_retriever(search_kwargs = {
        #                                        "vector_field": "content-embedding",
        #                                        "text_field": "content",
        #                                        "k": 5}),
        #                                    return_source_documents = True,
        #                                    memory = ConversationBufferMemory(input_key = "question",
        #                                                                      output_key = "answer",
        #                                                                      memory_key = "chat_history",
        #                                                                      return_messages = True),
        #                                    verbose = False)
        #msg = qa_chain({"query": prompt})["result"]
            memory = ConversationBufferMemory(memory_key='chat_history', return_messages=False)
            qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(), memory=memory)
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        msg = qa_chain.invoke({"question": prompt, "chat_history": chat_history})["answer"]
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar="📄").write(msg)

elif record_audio:
    def print_text():
        print(st.session_state.user_input)
    voice_prompt = get_asr(temp_audio_file)
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=print_text, key="user_input"):
    #if prompt := st.text_input(label="Text input", placeholder=voice_prompt, value=voice_prompt, key="user_input"):
#        st.session_state.user_input = voice_prompt 
        prompt=voice_prompt if prompt==' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if option == "anthropic.claude-v2:1":
            msg=bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)
        elif option == "gemini-pro":
            response=st.session_state.chat.send_message(prompt,stream=True,generation_config = gen_config)
            response.resolve()
            msg=response.text
        elif option == "gpt-4-1106-preview":
            msg=textGen(option, prompt, max_token, temperature, top_p)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar="🐶").write(msg)

else:
    if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            if option == "gemini-pro":
                response=st.session_state.chat.send_message(prompt,stream=True,generation_config = gen_config)
                response.resolve()
                msg=response.text
            elif option == "gpt-4-1106-preview":
                msg=textGen(option, prompt, max_token, temperature, top_p)
            elif option == "anthropic.claude-v2:1":
                msg=bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("ai", avatar="🐶").write(msg)
