import streamlit as st
from PIL import Image
import urllib.request
import typing
import os
import io
import http.client
import google.generativeai as genai
from vertexai.preview.generative_models import Part

def fetch_image_from_url(url:str):
    with urllib.request.urlopen(url) as url_response:
        # Read the image data from the URL response
        image_data = url_response.read()
        # Convert the image data to a BytesIO object
        image_stream = io.BytesIO(image_data)
        # Open the image using PIL
        return image_stream

st.set_page_config(page_title="Gemini Pro with Streamlit",page_icon="🩺")
google_api_key = os.getenv("gemini_api_token")
google_api_key = os.getenv("gemini_api_token")

genai.configure(api_key=google_api_key)

st.title("Gemini Pro with Streamlit Dashboard")

with st.sidebar:
    option = st.selectbox('Choose Your Model',('gemini-pro', 'gemini-pro-vision'))

    if 'model' not in st.session_state or st.session_state.model != option:
        st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
        st.session_state.model = option
    
    st.write("Adjust Your Parameter Here:")
    temperature = st.number_input("Temperature", min_value=0.0, max_value= 1.0, value =0.5, step =0.01)
    max_token = st.number_input("Maximum Output Token", min_value=0, value =100)
    gen_config = genai.types.GenerationConfig(max_output_tokens=max_token,temperature=temperature)

    #st.divider()
    #st.markdown("""<span ><font size=1>Connect With Me</font></span>""",unsafe_allow_html=True)
    #"[Linkedin](https://www.linkedin.com/in/cornellius-yudha-wijaya/)"
    #"[GitHub](https://github.com/cornelliusyudhawijaya)"
    
    st.divider()
    upload_images = st.file_uploader("Upload Your Images Here", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])
    image_url = st.text_input("Or Input Image URL", key="image_url", type="default")
    if upload_images:
        #image = Image.open(upload_image)
        for upload_file in upload_images:
            bytes_data = upload_file.read()
            image = Image.open(io.BytesIO(bytes_data))
    elif image_url:
        stream = fetch_image_from_url(image_url)
        st.image(stream, width=150)
        image = Image.open(stream)

    st.divider()
    upload_image2 = st.file_uploader("Upload the 2nd Images for comparison", accept_multiple_files=False, type=['jpg', 'jpeg', 'png'])
    if upload_image2:
        image2 = Image.open(upload_image2)

    st.divider()
    vedio_url = st.text_input("(TBD) Or Input Vedio URL with mp4 type", key="vedio_url", type="default")
    if vedio_url:
        vedio = Part.from_uri(
            uri= vedio_url,
            mime_type="video/mp4",
        )

    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

 
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if upload_images or image_url or vedio_url:
    if option == "gemini-pro":
        st.info("Please Switch to the Gemini Pro Vision")
        st.stop()
    if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            context = [prompt,image]
            if upload_image2:
                context = [prompt,image, image2]
            response=st.session_state.chat.send_message(context,stream=True,generation_config = gen_config)
            response.resolve()
            msg=response.text

            st.session_state.chat = genai.GenerativeModel(option).start_chat(history=[])
            st.session_state.messages.append({"role": "assistant", "content": msg})
            
            st.image(image,width=450)
            if upload_image2:
                st.image(image2,width=450)
            st.chat_message("assistant").write(msg)

else:
    if prompt := st.chat_input():
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            response=st.session_state.chat.send_message(prompt,stream=True,generation_config = gen_config)
            response.resolve()
            msg=response.text
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
    
    
