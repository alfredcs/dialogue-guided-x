## Using Amazon Teceract to extract texts, tables and forms and PyMuPDF to eatrct images froma pdf
from textractcaller.t_call import call_textract, Textract_Features
from trp import Document
import fitz
from PIL import Image
import os
import argparse
import glob
from operator import itemgetter
from itertools import groupby
import boto3
import json
import camelot
import matplotlib.pyplot as plt
import pandas as pd
import sys
from langchain.llms.bedrock import Bedrock
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

boto3_bedrock = bedrock.get_bedrock_client(
    #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)


def convert_pdf_image(filename: str):
    # Split the base name and extension
    output_directory_path, basename = os.path.splitext(filename)
    basename_n = basename.split('.')[0]
    if not os.path.exists(output_directory_path):
        os.makedirs(output_directory_path)
    
    # Open the PDF file
    doc = fitz.open(filename)
    try:
        # Iterate through each page and convert to an image
        for page_number in range(doc.page_count):
            # Get the page
            page = doc[page_number]
        
            # Convert the page to an image
            pix = page.get_pixmap()
        
            # Create a Pillow Image object from the pixmap
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
            # Save the image
            image.save(f"{output_directory_path}/{basename_n}/{page_number + 1}.png")
        
        # Close the PDF file
        doc.close()
        return f"{output_directory_path}/{basename_n}"
    except:
        print("Convert pdf to image failed!")
        return False

def isFloat(input):
  try:
    float(input)
  except ValueError:
    return False
  return True
            
def get_tables(pdf_filename:str):
    image_dir = convert_pdf_image(pdf_filename)
    warning = ""
    texts = tables = forms = []
    for file in glob.glob(os.path.join(image_dir, "*")):
        with open(file, 'rb') as document:
            imageBytes = bytearray(document.read())
        
        # Call Amazon Textract
        table_response = call_textract(input_document=imageBytes, features=[Textract_Features.TABLES])
        form_response = call_textract(input_document=imageBytes, features=[Textract_Features.FORMS])
        text_response = call_textract(input_document=imageBytes)
    
        doc = Document(table_response)
        for page in doc.pages:
            table_text = ''
            for table in page.tables:
                for r, row in enumerate(table.rows):
                    itemName  = ""
                    for c, cell in enumerate(row.cells):
                        table_text += f"Table[{r}][{c}] = {cell.text}, "
                        if(c == 0):
                            itemName = cell.text
                        elif(c == 4 and isFloat(cell.text)):
                            value = float(cell.text)
                            if(value > 1000):
                                warning += "{} is greater than $1000.".format(itemName)
                tables.append(table_text[:len(table_text)-2])
        document.close() 
        # Comment out if needed
        doc = Document(form_response)
        for page in doc.pages:
            form_text = ''
            for table in page.tables:
                for r, row in enumerate(table.rows):
                    itemName  = ""
                    for c, cell in enumerate(row.cells):
                        form_text += f"Table[{r}][{c}] = {cell.text}, "
                        if(c == 0):
                            itemName = cell.text
                        elif(c == 4 and isFloat(cell.text)):
                            value = float(cell.text)
                            if(value > 1000):
                                warning += "{} is greater than $1000.".format(itemName)
                forms.append(form_text[:len(form_text)-2])
        document.close() 
        doc = Document(text_response)
        for page in doc.pages:
            text = ''
            for line in page.lines:
                text += f"{line.text}, "
        texts.append(text[:len(text)-2])
        document.close() 

    return texts, tables, forms


def get_images(filename: str):
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename).split('.')[0]
    new_dir = f"{dirname}/{basename}"
    os.mkdir(new_dir ) if not os.path.exists(new_dir ) else None  
    doc = fitz.open(filename) # open a document

    try:
        for page_index in range(len(doc)): # iterate over pdf pages
        	page = doc[page_index] # get the page
        	image_list = page.get_images()
        
        	# print the number of images found on the page
        	if image_list:
        		print(f"Found {len(image_list)} images on page {page_index}")
        	else:
        		print("No images found on page", page_index)
        
        	for image_index, img in enumerate(image_list, start=1): # enumerate the image list
        		xref = img[0] # get the XREF of the image
        		pix = fitz.Pixmap(doc, xref) # create a Pixmap
        
        		if pix.n - pix.alpha > 3: # CMYK: convert to RGB first
        			pix = fitz.Pixmap(fitz.csRGB, pix)
        
        		pix.save(f"{new_dir}/%s-page_%s-image_%s.png" % (os.path.splitext(os.path.basename(filename))[0], page_index, image_index)) # save the image as png
        		pix = None
        return new_dir 
    except:
        print("EXtract image from pdf failed!")
        return False

# ==============================================================================
# Function ParseTab - parse a document table into a Python list of lists
# ==============================================================================
def ParseTab(page, bbox, columns=None):
    """Returns the parsed table of a page in a PDF / (open) XPS / EPUB document.
    Parameters:
    page: fitz.Page object
    bbox: containing rectangle, list of numbers [xmin, ymin, xmax, ymax]
    columns: optional list of column coordinates. If None, columns are generated
    Returns the parsed table as a list of lists of strings.
    The number of rows is determined automatically
    from parsing the specified rectangle.
    """
    tab_rect = fitz.Rect(bbox).irect
    xmin, ymin, xmax, ymax = tuple(tab_rect)

    if tab_rect.is_empty or tab_rect.is_infinite:
        print("Warning: incorrect rectangle coordinates!")
        return []

    if type(columns) is not list or columns == []:
        coltab = [tab_rect.x0, tab_rect.x1]
    else:
        coltab = sorted(columns)

    if xmin < min(coltab):
        coltab.insert(0, xmin)
    if xmax > coltab[-1]:
        coltab.append(xmax)

    words = page.get_text("words")

    if words == []:
        print("Warning: page contains no text")
        return []

    alltxt = []

    # get words contained in table rectangle and distribute them into columns
    for w in words:
        ir = fitz.Rect(w[:4]).irect  # word rectangle
        if ir in tab_rect:
            cnr = 0  # column index
            for i in range(1, len(coltab)):  # loop over column coordinates
                if ir.x0 < coltab[i]:  # word start left of column border
                    cnr = i - 1
                    break
            alltxt.append([ir.x0, ir.y0, ir.x1, cnr, w[4]])

    if alltxt == []:
        print("Warning: no text found in rectangle!")
        return []

    alltxt.sort(key=itemgetter(1))  # sort words vertically

    # create the table / matrix
    spantab = []  # the output matrix

    for y, zeile in groupby(alltxt, itemgetter(1)):
        schema = [""] * (len(coltab) - 1)
        for c, words in groupby(zeile, itemgetter(3)):
            entry = " ".join([w[4] for w in words])
            schema[c] = entry
        spantab.append(schema)

    return spantab


def save_table_as_image(table, image_path):
    """
    Saves a table (pandas DataFrame) as an image.
    """
    # Create a figure and a single subplot
    fig, ax = plt.subplots(figsize=(table.shape[1], table.shape[0]))  # Size might need adjustment
    ax.axis('tight')
    ax.axis('off')
    table_ax = ax.table(cellText=table.values, colLabels=table.columns, loc='center')
    plt.savefig(image_path)
    plt.close(fig)

def extract_tables_and_save_as_images(pdf_path, output_dir):
    """
    Extracts tables from a PDF file and saves each table as an image.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract tables from the PDF
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')

    # Loop through the extracted tables
    for i, table in enumerate(tables):
        # Convert the table to a pandas DataFrame
        df = table.df

        # Define the path for the output image
        image_path = os.path.join(output_dir, f'table_{i}.png')

        # Save the DataFrame as an image
        save_table_as_image(df, image_path)

        print(f"Table {i} saved as image at {image_path}")

#-----
def save_table_as_image(table, image_path):
    """
    Saves a table (DataFrame) as an image using Matplotlib.
    """
    fig, ax = plt.subplots(figsize=(16, 10))  # Adjust the size as necessary
    ax.axis('off')
    table_ax = ax.table(cellText=table.values, colLabels=table.columns, loc='center', cellLoc='center')
    table_ax.auto_set_font_size(False)
    table_ax.set_fontsize(12)  # Adjust the font size as necessary
    plt.savefig(image_path, bbox_inches='tight', dpi=120)
    plt.close()

def extract_and_save_tables_as_images(pdf_path, output_dir):
    """
    Extracts tables from a PDF file using Camelot and saves each table as an image.
    """
    basename_n = os.path.basename(pdf_path).split('.')[0]
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract tables from the PDF
    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')  # Try 'lattice' if 'stream' doesn't work well

    # Loop through the extracted tables
    tables_list = []
    for i, table in enumerate(tables):
        # Convert the table to a pandas DataFrame
        df = table.df

        # Define the path for the output image
        image_path = os.path.join(output_dir, f'{basename_n}_table_{i+1}.png')

        # Save the DataFrame as an image
        save_table_as_image(df, image_path)

        print(f"Table {i+1} saved as image at: {image_path}")

        #Convert to json
        json_str = df.to_json(orient='records')

        tables_list.append(json_str)

        # Save the JSON to a file
        with open(f'{output_dir}/{basename_n}_table_{i+1}.json', 'w') as file:
            file.write(json_str)
        # Save the CSV to a file
        df.to_csv(f'{output_dir}/{basename_n}_table_{i+1}.csv', index=False)
        
    return tables_list


#-----
def text_summary_from_json(json_input:str):
    model = Bedrock(
        model_id="anthropic.claude-v2", 
        client=boto3_bedrock,
        model_kwargs={'temperature': 0.3, 'max_tokens_to_sample': 1024}
    )
    
    str_parser = StrOutputParser()

    prompt = PromptTemplate(
        template="""     
        Human:
        {instructions} : \"{document}\"
        Assistant:""",
        input_variables=["instructions","document"]
    )
    
    summary_chain = prompt | model | StrOutputParser()
    response = summary_chain.invoke({
        "instructions":"You will be provided with multiple sets of insights to describe the table based on the following json format. Compile and summarize columns and rows to capture all key components with comprehensive and accuract text description of the table. Form your summary in one paragraph",
        "document": {'\n'.join(json_input)}
    })
    return response
    
#----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process command line input')
    parser.add_argument('-f', '--pdf_files', nargs="*", type=str, help='PDF s files in list format')
    parser.add_argument('-o', '--output_path', type=str, help='Output directory')
    args = parser.parse_args()
    pdf_file_list = args.pdf_files
    output_dir = args.output_path
    bucket = 'sagemaker-us-west-2-415275363822'
    for f in args.pdf_files:
        tables = extract_and_save_tables_as_images(f, output_dir)
        #extract_tables_and_save_as_images(f, output_dir)
        #tables = extract_tables_textract(f, output_dir)
        #get_images(f)
        #_, tables, _= get_tables(f)
        print(tables)
        for table in tables:
            text = text_summary_from_json(table)
            print(text)
        '''
        #tabs = parse_tables(f)
        #df = tables[0].to_pandas
        print(tabs.tables)
        for i,tab in enumerate(tabs):  # iterate over all tables
            for cell in tab.header.cells:
                page.draw_rect(cell,color=fitz.pdfcolor["red"],width=0.3)
            page.draw_rect(tab.bbox,color=fitz.pdfcolor["green"])
            print(f"Table {i} column names: {tab.header.names}, external: {tab.header.external}")
        '''
            

    
    