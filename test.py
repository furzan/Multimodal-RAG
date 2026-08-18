from unstructured.partition.pdf import partition_pdf
import os

import unstructured_pytesseract
unstructured_pytesseract.pytesseract.tesseract_cmd = r'D:\s\tesseract\New folder\tesseract.exe'
os.environ["PATH"] += os.pathsep + r'D:\s\poppler\New folder\poppler-26.02.0\Library\bin'

output_path = "./content/"
file_path = output_path + 'diabetes.pdf'

# Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
chunks = partition_pdf(
    filename="./backend/app/uploads/DIABETES.pdf",
    infer_table_structure=True,            # extract tables
    strategy="hi_res",                     # mandatory to infer tables

    extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
    # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

    extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

    chunking_strategy="by_title",          # or 'basic'
    max_characters=10000,                  # defaults to 500
    combine_text_under_n_chars=2000,       # defaults to 0
    new_after_n_chars=6000,

    # extract_images_in_pdf=True,          # deprecated
)


print(f"\n--- Processing Complete! Found {len(chunks)} elements ---\n")

# Print the text of the first 5 elements to test it out
for i, chunk in enumerate(chunks[:5]):
    print(f"--- Element {i+1} ({type(chunk).__name__}) ---")
    print(chunk.text)
    print("-" * 30)