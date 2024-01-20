# mm-Rag helper
import os
import base64
import io, sys, boto3
import fitz  # PyMuPDF
from io import BytesIO
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium
from langchain_openai import (OpenAIEmbeddings, ChatOpenAI)
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain.schema.messages import HumanMessage
import uuid
import getpass
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.storage import InMemoryStore
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import (Chroma, OpenSearchVectorSearch)
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_benchmarks.rag.tasks.multi_modal_slide_decks import get_file_names
from langchain_benchmarks import clone_public_dataset
from langchain_benchmarks import registry as RY
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

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

def get_base64_encoded_images_from_dir(directory_path):
    """
    Read all image files in a directory and encode them into a list of Base64 strings.

    :param directory: The directory containing the image files.
    :return: A list of Base64 encoded strings representing the images.
    """
    # List to hold Base64 encoded strings
    base64_images = []

    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

    # Iterate through all files in the directory
    file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    for filename in file_names:
        # Check if the file is an image
        if filename.lower().endswith(valid_extensions):
            # Construct full file path
            file_path = os.path.join(directory_path, filename)
            
            # Open the image, convert it to bytes and encode in Base64
            try:
                with Image.open(file_path) as img:
                    buffered = BytesIO()
                    img.save(buffered, format=img.format)
                    img_bytes = buffered.getvalue()
                    img_base64 = base64.b64encode(img_bytes)
                    base64_images.append(img_base64.decode('utf-8'))
            except Exception as e:
                print(f'Error processing image {filename}: {e}')

    return base64_images


def extract_images_from_pdf(pdf_file):
    """
    Extract images from a PDF and save them in a specified directory.

    :param pdf_path: Path to the PDF file.
    :param output_dir: Directory to save the extracted images.
    """
    # Check if the output directory exists, create if not
    output_dir =  os.path.dirname(pdf_file)+'/images'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Open the PDF file
    document = fitz.open(pdf_file)

    pil_images = []

    # Iterate through each page
    for page_num in range(len(document)):
        # Get the page
        page = document.load_page(page_num)

        # Get the image list of the page
        image_list = page.get_images(full=True)

        # Iterate through each image in the list
        for image_index, img in enumerate(image_list, start=1):
            # Get the XREF of the image
            xref = img[0]
            # Extract the image bytes
            base_image = document.extract_image(xref)
            image_bytes = base_image["image"]
            # Construct the image path
            image_path = os.path.join(output_dir, f"{os.path.basename(pdf_file)}_{page_num+1}_{image_index}.{base_image['ext']}")
            # Save the image
            with open(image_path, "wb") as image_file:
                image_file.write(image_bytes)

            pil_images.append(image_bytes)

    # Close the document
    document.close()
    return output_dir, pil_images


def get_images(file):
    """
    Get PIL images from PDF pages and save them to a specified directory
    :param file: Path to file
    :return: A list of PIL images
    """

    # Get presentation
    pdf = pdfium.PdfDocument(file)
    n_pages = len(pdf)

    # Extracting file name and creating the directory for images
    file_name = Path(file).stem  # Gets the file name without extension
    img_dir = os.path.join(Path(file).parent, "img")
    os.makedirs(img_dir, exist_ok=True)

    # Get images
    pil_images = []
    print(f"Extracting {n_pages} images for {file_name}")
    for page_number in range(n_pages):
        page = pdf.get_page(page_number)
        bitmap = page.render(scale=1, rotation=0, crop=(0, 0, 0, 0))
        pil_image = bitmap.to_pil()
        pil_images.append(pil_image)

        # Saving the image with the specified naming convention
        image_path = os.path.join(img_dir, f"{file_name}_image_{page_number + 1}.jpg")
        pil_image.save(image_path, format="JPEG")

    return pil_images

def resize_base64_image(base64_string, size=(128, 128)):
    """
    Resize an image encoded as a Base64 string

    :param base64_string: Base64 string
    :param size: Image size
    :return: Re-sized Base64 string
    """
    # Decode the Base64 string
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))

    # Resize the image
    resized_img = img.resize(size, Image.LANCZOS)

    # Save the resized image to a bytes buffer
    buffered = io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # Encode the resized image to Base64
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    img_str = resize_base64_image(img_str, size=(960, 540))
    return img_str


def image_summarize(img_base64, prompt):
    """
    Make image summary

    :param img_base64: Base64 encoded string for image
    :param prompt: Text prompt for summarizatiomn
    :return: Image summarization prompt

    """
    chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)

    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content

def generate_img_summaries(img_base64_list):
    """
    Generate summaries for images

    :param img_base64_list: Base64 encoded images
    :return: List of image summaries and processed images
    """

    # Store image summaries
    image_summaries = []
    processed_images = []

    # Prompt
    prompt = """You are an assistant tasked with summarizing images for retrieval. \
    These summaries will be embedded and used to retrieve the raw image. \
    Please take a moment to thoroughly examine the image, paying close attention to both the overarching theme and the finer details, before providing your response. \
    Give a concise summary of the image that is well optimized for retrieval."""

    # Apply summarization to images
    for i, base64_image in enumerate(img_base64_list):
        try:
            image_summaries.append(image_summarize(base64_image, prompt))
            processed_images.append(base64_image)
        except:
            print(f"BadRequestError with image {i+1}")

    return image_summaries, processed_images

def create_multi_vector_retriever(vectorstore, image_summaries, images):
    """
    Create retriever that indexes summaries, but returns raw images or texts

    :param vectorstore: Vectorstore to store embedded image sumamries
    :param image_summaries: Image summaries
    :param images: Base64 encoded images
    :return: Retriever
    """

    # Initialize the storage layer
    store = InMemoryStore()
    id_key = "doc_id"

    # Create the multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    # Helper function to add documents to the vectorstore and docstore
    def add_documents(retriever, doc_summaries, doc_contents):
        doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(doc_summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, doc_contents)))

    add_documents(retriever, image_summaries, images)
    return retriever

def prepare_images(docs):
    """
    Prepare iamges for prompt

    :param docs: A list of base64-encoded images from retriever.
    :return: Dict containing a list of base64-encoded strings.
    """
    b64_images = []
    for doc in docs:
        if isinstance(doc, Document):
            doc = doc.page_content
        b64_images.append(doc)
    return {"images": b64_images}


def img_prompt_func(data_dict, num_images=2):
    """
    GPT-4V prompt for image analysis.

    :param data_dict: A dict with images and a user-provided question.
    :param num_images: Number of images to include in the prompt.
    :return: A list containing message objects for each image and the text prompt.
    """
    messages = []
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"][:num_images]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)
    text_message = {
        "type": "text",
        "text": (
            "You are an analyst tasked with answering questions about visual content.\n"
            "You will be give a set of image(s) from a slide deck / presentation.\n"
            "Use this information to answer the user question. \n"
            f"User-provided question: {data_dict['question']}\n\n"
        ),
    }
    messages.append(text_message)
    return [HumanMessage(content=messages)]


def multi_modal_rag_chain(retriever):
    """
    Multi-modal RAG chain
    """

    # Multi-modal LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)

    # RAG pipeline
    chain = (
        {
            "context": retriever | RunnableLambda(prepare_images),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(img_prompt_func)
        | model
        | StrOutputParser()
    )

    return chain

def do_mmrag(directory_path):
    '''
    # Extract data from dataset
    registry = RY.filter(Type="RetrievalTask")
    task = registry["Multi-modal slide decks"]
    clone_public_dataset(task.dataset_id, dataset_name=task.name)

    file_names = list(get_file_names())  # PosixPath

    file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    # Extract images from odf
    images = []
    for fi in file_names:
        print(f'{directory_path}/{fi}')
        #images.extend(get_images(directory_path+'/'+fi))
        images.extend( extract_images_from_pdf(directory_path+'/'+fi))
    '''
    
    # Extract images from pdf
    file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    images = []
    for fi in file_names:
        images_path, images = extract_images_from_pdf(directory_path+'/'+fi)
    # Covert to base64
    #images_base_64 = [convert_to_base64(i) for i in images]
    images_base_64 = get_base64_encoded_images_from_dir(images_path)

    #choose mebedding engine
    embedding_openCLIP = OpenCLIPEmbeddings()
    embedding_bedrock = BedrockEmbeddings(client=boto3_bedrock, model_id="amazon.titan-embed-image-v1:0")

    # Make vectorstore
    vectorstore_mmembd = Chroma(
        collection_name="multi-modal-rag", embedding_function=embedding_openCLIP
    )

    # Use AOSS
    collection_name = 'bedrock-workshop-rag'
    collection_id = '967j1chec55256z804lj'
    aoss_host = "{}.{}.aoss.amazonaws.com:443".format(collection_id, my_region)
    aoss_mmembd = OpenSearchVectorSearch(index_name=collection_name, 
                                                 embedding_function=embedding_bedrock, 
                                                 opensearch_url=aoss_host,
                                                 http_auth=auth,
                                                 timeout = 100,
                                                 use_ssl = True,
                                                 verify_certs = True,
                                                 connection_class = RequestsHttpConnection,
                                                 is_aoss=True,
    )
    
    # Read images we extracted above
    image_uris  = [images_path+'/'+f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    image_uris.sort()
    # Add images
    vectorstore_mmembd.add_images(uris=image_uris)
       
    # Make retriever
    retriever_mmembd = vectorstore_mmembd.as_retriever()

    # Image summaries
    image_summaries, images_base_64_processed = generate_img_summaries(images_base_64)
    # Insert image summaries into aoss
    #aoss_mmembd.add_texts(texts=image_summaries) 
    
    # The vectorstore to use to index the summaries
    vectorstore_mvr = Chroma(
        collection_name="multi-modal-rag-mv", embedding_function=embedding_bedrock
    )
    
    # Create retriever
    retriever_multi_vector_img = create_multi_vector_retriever(
        vectorstore_mvr,
        image_summaries,
        images_base_64_processed,
    )
    # Create RAG chain
    chain_multimodal_rag = multi_modal_rag_chain(retriever_multi_vector_img)
    chain_multimodal_rag_mmembd = multi_modal_rag_chain(retriever_mmembd)

    # Image summaries
    image_summaries, images_base_64_processed = generate_img_summaries(images_base_64)
    return image_summaries, images_base_64_processed 

if __name__ == "__main__":
    #os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    #env_vars = ["LANGCHAIN_API_KEY", "OPENAI_API_KEY"]
    #env_vars = ["OPENAI_API_KEY"]
    #for var in env_vars:
    #    if var not in os.environ:
    #        os.environ[var] = getpass.getpass(prompt=f"Enter your {var}: ")
    image_summaries, images_base_64_processed  = do_mmrag('./pdf_temp2')
    print(image_summaries)
