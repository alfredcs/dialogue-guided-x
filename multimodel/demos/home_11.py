# Version: v0.05: adding RAG solution without MM
#
import streamlit as st
from PIL import Image
import urllib.request
import typing
import os
import sys
import io
from operator import itemgetter
import google.generativeai as genai
#from vertexai.preview.generative_models import (GenerativeModel, Part)
from openai import OpenAI
import base64
import requests
from io import BytesIO
import urllib.request
from brain import get_index_for_pdf
import hmac
#from streamlit_mic_recorder import mic_recorder, speech_to_text
from audiorecorder import audiorecorder




module_path = ".."
sys.path.append(os.path.abspath(module_path))
from claude_bedrock import bedrock_textGen, bedrock_llm, bedrock_imageGen, bedrock_textGen_agent, prompt_rewrite
from rad_tools import *
from utils.gemini_generative_models import _GenerativeModel as GenerativeModel
from utils.gemini_generative_models import Part 

st.set_page_config(page_title="MM-RAG Demo",page_icon="🩺")
st.title("Multimodal RAG Demo")
video_file_name = "download_video.mp4"
client = OpenAI(api_key=os.getenv('openai_api_token'))
voice_prompt = ""
chat_history = []
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

## Image generation prompt rewrite
prefix = "Your role as an expert prompt engineer involves accuratly and meticulously rewriting the input text without altering original meaning, transforming it into a precised, detailed and enriched text prompt. This refined prompt is destined for a text-to-image generation model. Your primary objective is to strickly and precisely maintain the key elements and core semantic essence of the original text while infusing it with rich, descriptive elements. Such detailed guidance is crucial for steering the image generation model towards producing images of superior quality, characterized by their vivid and expressive visual nature. Your adeptness in prompt crafting is instrumental in ensuring that the final images not only captivate visually but also resonate deeply with the original textual concept. Please rewrite this prompt: "

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
    output = response.text.rstrip()
    if output == "Thank you." or output == "Bye.":
        return ""
    else:
        return output

# ++++++++++++++ Local deployed models with TGI>=1.4 ++++++++++++++++++++++
    
    

# Check password
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

with st.sidebar:
    st.title(':orange[Multimodal Config] :pencil2:')
    option = st.selectbox('Choose Model',('anthropic.claude-v2:1', 'mistral-7b', 'gemini-pro', 'gemini-pro-vision', 'gpt-4-1106-preview', 'gpt-4-vision-preview', 'stability.stable-diffusion-xl-v1:0', 'amazon.titan-image-generator-v1', 'llava-v1.5-13b-vision', 'dall-e-3'))

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
    text_embedding_option = st.selectbox('Choose Embedding Model',('titan', 'openai', 'hf-tei'))
    st.write("------- Image Generation----------")
    image_n = st.number_input("Choose number of images", min_value=1, value=1, max_value=1)
    image_size = st.selectbox('Choose image size', ('1024x1024', '1024x1792', '1792x1024'))
    image_quality = st.selectbox('Choose image quality', ('standard', 'hd'))
    cfg =  st.number_input("Choose CFG Scale for freedom", min_value=1.0, value=7.5, max_value=15.0)
    seed = st.slider('Choose seed for noise pattern', -1, 214783647, 452345)

    st.divider()
    st.header(':green[Multimodal RAG] :file_folder:')
    upload_docs = st.file_uploader("Upload pdf files", accept_multiple_files=True, type=['pdf'])
    doc_urls = st.text_input("Or input URLs seperated by ','", key="doc_urls", type="default")
    if text_embedding_option == 'titan':
        text_embedding = BedrockEmbeddings(client=boto3_bedrock, model_id="amazon.titan-embed-text-v1")
        chunk_size = 8000
    elif text_embedding_option == 'openai':
        text_embedding =  OpenAIEmbeddings(openai_api_key=os.getenv('openai_api_token'))
        chunk_size = 8000
    elif text_embedding_option == 'hf-tei':
        text_embedding = HuggingFaceHubEmbeddings(model='http://infs.cavatar.info:8084')
        chunk_size = 500

    if upload_docs:
        upload_doc_names = [file.name for file in upload_docs]
        for upload_doc in upload_docs:
            bytes_data = upload_doc.read()
            with open(upload_doc.name, 'wb') as f:
                f.write(bytes_data)
        #st.session_state["vectordb"] = create_vectordb(upload_docs, upload_doc_names)
        docs, avg_doc_length = data_prep(upload_doc_names,chunk_size=chunk_size)
        #print(f'Docs:{upload_doc_names} and avg sizes:{avg_doc_length}')
        vlen = update_vdb(docs, text_embedding, aoss_host, collection_name, profile_name, my_region)
        msg = f'Total {vlen} papges of document was added to vectorDB.'
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai").write(msg)
    if doc_urls:
        docs, avg_doc_length = data_prep(doc_urls.split(","), chunk_size=chunk_size)
        vlen = update_vdb(docs, text_embedding, aoss_host, collection_name, profile_name, my_region)
        msg = f'Total {vlen} papges of document was added to vectorDB.'
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai").write(msg)
    st.caption('Only tested with Claude and GPT-4')
    rag_on = st.toggle('Activate RAG retrival')
    
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
        try:
            stream = fetch_image_from_url(image_url)
            st.image(stream)
            image = Image.open(stream)
        except:
            msg = 'Failed to download image, please check permission.'
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("ai").write(msg)

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

    #st.header(':green[Audio understanding] :microphone:')
    #sample_audio = st.file_uploader("Upload a sample audio file for cloning", type=["mp3", "wav"], accept_multiple_files=False)
    #if sample_audio:
    #    sample_audio_bytes = sample_audio.read()
    #    st.audio(sample_audio_bytes, format="audio/wav")#, start_time=0, *, sample_rate=None)
    ##st.caption('Multilingual transcribe')
    ##record_audio=mic_recorder(start_prompt="▶️ ", stop_prompt="⏹️" ,key='recorder')

    #st.divider()
    st.header(':green[Enable Agent with voice] :microphone:')
    record_audio=audiorecorder(start_prompt="Voice input start:  ▶️ ", stop_prompt="Record stop: ⏹️", pause_prompt="", key=None)
    if len(record_audio)>3:
        record_audio_bytes = record_audio.export().read()
        st.audio(record_audio_bytes, format="audio/wav")#, start_time=0, *, sample_rate=None)
        record_audio.export(temp_audio_file, format="mp3")
        if os.path.exists(temp_audio_file):
            voice_prompt = get_asr(temp_audio_file)
            #os.remove(temp_audio_file)
        record_audio.empty()
        #if os.path.exists(temp_audio_file):
        #    os.remove(temp_audio_file)
    st.caption("Press space and hit ↩️ for voice & agent activation")
    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]
        record_audio = None
        #del st.session_state[record_audio]

# =============================================

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "I am your assistant. How can I help today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if upload_images or image_url:
    if option != "gemini-pro-vision" and option != "gpt-4-vision-preview" and option != 'llava-v1.5-13b-vision' and option != 'amazon.titan-image-generator-v1':
        st.info("Please switch to a vision model")
        st.stop()
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt=voice_prompt if prompt==' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        print(f'In image on {option}')
        try:
            if option == "gemini-pro-vision":
                context = [prompt,image]
                #context = [prompt, image] if upload_images or image_url else ([prompt, video] if video_url else None)
                if upload_image2:
                    context = [prompt,image, image2]
                response=st.session_state.chat.send_message(context,stream=True,generation_config = gen_config)
                response.resolve()
                msg=response.text
            elif option == 'llava-v1.5-13b-vision' and image_url:
                image_prompt = f'({image_url}) {prompt}'
                msg=tgi_imageGen('http://infs.cavatar.info:8085/generate', image_prompt, max_token, temperature, top_p, top_k)
            elif option == "gpt-4-vision-preview":
                msg = getDescription(option, prompt, image, max_token, temperature, top_p)
                if upload_image2:
                     msg = getDescription2(option, prompt, image, image2, max_token, temperature, top_p)
            elif option == "amazon.titan-image-generator-v1":
                new_prompt = tgi_textGen('http://infs.cavatar.info:8080', f'{prefix} {prompt}', max_token, temperature, top_p, top_k)
                src_image = image if 'image' in locals() else None
                image_quality = 'premium' if image_quality == 'hd' else image_quality
                try:
                    base64_str = bedrock_imageGen(option, new_prompt, iheight=1024, iwidth=1024, src_image=image, image_quality=image_quality, image_n=image_n, cfg=cfg, seed=seed)
                    new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
                    st.image(new_image,use_column_width='auto')
                    msg = ' '
                except:
                    msg = 'Server error encountered. Please try again later!.'
                    pass
            else:
                msg = "Please choose a correct model."
        except:
            msg = "Server error encountered. Please try again later."
            pass
        msg += "\n\n✒︎Content created by using: " + option
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.messages.append({"role": "assistant", "content": msg})
        
        st.image(image)
        if upload_image2:
            st.image(image2)
        st.chat_message("assistant", avatar='🌈').write(msg)
elif upload_video or video_url:
    if option != "gemini-pro-vision" and option != "gpt-4-vision-preview" and option != 'llava-v1.5-13b-vision':
        st.info("Please switch to a vision model")
        st.stop()
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        try:
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
            elif option == 'llava-v1.5-13b-vision' and  image_url:
                image_prompt = f'({image_url}){prompt}'
                msg=tgi_imageGen('http://infs.cavatar.info:8085/generate', image_prompt, max_token, temperature, top_p, top_k)
        except:
            msg = "Server error encountered. Please try again."
            pass
        msg += "\n\n✒︎Content created by using: " + option
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.messages.append({"role": "assistant", "content": msg})

        #video_file = open(video_file_name, 'rb')
        #video_bytes = video_file.read()
        st.video(video_bytes, start_time=0)
        st.chat_message("assistant", avatar='🎞️').write(msg)
elif rag_on:
    if option == "gemini-pro-vision" or option == "gpt-4-vision-preview" or option == "llava-v1.5-13b-vision":
        st.info("Please switch to a text model")
        st.stop()
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt=voice_prompt if prompt==' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        try:
            if text_embedding_option == 'titan':
                text_embedding = BedrockEmbeddings(client=boto3_bedrock, model_id="amazon.titan-embed-text-v1")
            elif text_embedding_option == 'openai':
                text_embedding =  OpenAIEmbeddings(openai_api_key=os.getenv('openai_api_token'))
            elif text_embedding_option == 'hf-tei':
                 ext_embedding = HuggingFaceHubEmbeddings(model='http://infs.cavatar.info:8084')

            #print(f'RAG:{prompt}, model to use:{option}')
            msg = do_query(prompt, option, text_embedding, aoss_host, collection_name, profile_name, max_token, temperature, top_p, top_k, my_region)
        except:
            msg = "Server error encountered. Please try again."
            pass
        msg += "\n\n✒︎Content created by using: RAG with " + option
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar="📄").write(msg)

elif (record_audio and len(voice_prompt) > 1):
    if prompt := st.chat_input(placeholder=voice_prompt, on_submit=None, key="user_input"):
        prompt=voice_prompt if prompt==' ' else prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        try:
            if option == "anthropic.claude-v2:1":
                msg=bedrock_textGen_agent(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)
            elif option == "gemini-pro":
                response=st.session_state.chat.send_message(prompt,stream=True,generation_config = gen_config)
                response.resolve()
                msg=response.text
            elif option == "gpt-4-1106-preview":
                msg=textGen_agent(option, prompt, max_token, temperature, top_p)
            elif option == "mistral-7b":
                msg=tgi_textGen('http://infs.cavatar.info:8080', prompt, max_token, temperature, top_p, top_k)
            elif option == 'llava-v1.5-13b-vision' and image_url:
                msg=tgi_textGen('http://infs.cavatar.info:8085', prompt, max_token, temperature, top_p, top_k)
            elif option == "amazon.titan-image-generator-v1" or option == "stability.stable-diffusion-xl-v1:0":
                src_image = image if 'image' in locals() else None
                image_quality = 'premium' if image_quality == 'hd' else image_quality
                new_prompt =tgi_textGen('http://infs.cavatar.info:8080', f'{prefix} {prompt}', max_token, temperature, top_p, top_k)
                base64_str = bedrock_imageGen(option, new_prompt, iheight=1024, iwidth=1024, src_image=src_image, image_quality=image_quality, image_n=image_n, cfg=cfg, seed=seed)
                new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
                st.image(new_image,use_column_width='auto')
                msg = new_prompt
            else:
                msg = "Please choose a correct model."
        except:
            msg = "Server error encountered. Please try again later."
            pass
        msg += "\n\n✒︎Content created by using: LangChain Agent and " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🎙️').write(msg)

else:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        try:
            if option == "gemini-pro":
                response=st.session_state.chat.send_message(prompt,stream=True,generation_config = gen_config)
                response.resolve()
                msg=response.text
            elif option == "gpt-4-1106-preview":
                msg=textGen(option, prompt, max_token, temperature, top_p)
            elif option == "anthropic.claude-v2:1":
                msg=bedrock_textGen(option, prompt, max_token, temperature, top_p, top_k, stop_sequences)
            elif option == "mistral-7b":
                msg=tgi_textGen('http://infs.cavatar.info:8080', prompt, max_token, temperature, top_p, top_k)
            elif option == 'llava-v1.5-13b-vision':
                msg=tgi_textGen('http://infs.cavatar.info:8085', prompt, max_token, temperature, top_p, top_k)
            elif option == "amazon.titan-image-generator-v1" or option == "stability.stable-diffusion-xl-v1:0":
                src_image = image if 'image' in locals() else None
                image_quality = 'premium' if image_quality == 'hd' else image_quality
                base64_str = bedrock_imageGen(option, prompt, iheight=1024, iwidth=1024, src_image=src_image, image_quality=image_quality, image_n=image_n, cfg=cfg, seed=seed)
                new_image = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))
                st.image(new_image,use_column_width='auto')
                msg = ' '
            elif option == "dall-e-3":
                image_url = dalle3_imageGen(option, prompt, size=image_size, quality=image_quality, n_image=image_n)
                st.image(image_url)
                msg = ''
            else:
                msg = "Please choose a correct model."
        except Exception as err:
            msg = "Server error encountered. Please try again later." 
            pass
        msg += "\n\n ✒︎Content created by using: " + option
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("ai", avatar='🦙').write(msg)
