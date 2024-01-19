# mm-Rag helper
import os
import base64
import io
from io import BytesIO
from pathlib import Path
from PIL import Image
import pypdfium2 as pdfium
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
import uuid

from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.schema.output_parser import StrOutputParser
from langchain.storage import InMemoryStore
from langchain.vectorstores import Chroma
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_benchmarks.rag.tasks.multi_modal_slide_decks import get_file_names


def get_images_from_pdf(file):
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
    print(f"Extracting {n_pages} images for {file.name}")
    for page_number in range(n_pages):
        page = pdf.get_page(page_number)
        bitmap = page.render(scale=1, rotation=0, crop=(0, 0, 0, 0))
        pil_image = bitmap.to_pil()
        pil_images.append(pil_image)

        # Saving the image with the specified naming convention
        image_path = os.path.join(img_dir, f"{file_name}_image_{page_number + 1}.jpg")
        pil_image.save(image_path, format="JPEG")

    return pil_images

ef resize_base64_image(base64_string, size=(128, 128)):
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

ef prepare_images(docs):
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

def do_mmrag():

    # Extract data from dataset
    registry = registry.filter(Type="RetrievalTask")
    task = registry["Multi-modal slide decks"]
    clone_public_dataset(task.dataset_id, dataset_name=task.name)

    file_names = list(get_file_names())  # PosixPath
    
    # Extract images from odf
    images = []
    for fi in file_names:
        images.extend(get_images(fi))

    # Covert to base64
    images_base_64 = [convert_to_base64(i) for i in images]

    # Make vectorstore
    vectorstore_mmembd = Chroma(
        collection_name="multi-modal-rag",
        embedding_function=OpenCLIPEmbeddings(),
    )
    
    # Read images we extracted above
    img_dir = os.path.join(Path(file_names[0]).parent, "img")
    image_uris = sorted(
        [
            os.path.join(img_dir, image_name)
            for image_name in os.listdir(img_dir)
            if image_name.endswith(".jpg")
        ]
    )

    # Add images
    vectorstore_mmembd.add_images(uris=image_uris)
    
    # Make retriever
    retriever_mmembd = vectorstore_mmembd.as_retriever()

    # Image summaries
    image_summaries, images_base_64_processed = generate_img_summaries(images_base_64)

    # The vectorstore to use to index the summaries
    vectorstore_mvr = Chroma(
        collection_name="multi-modal-rag-mv", embedding_function=OpenAIEmbeddings()
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
