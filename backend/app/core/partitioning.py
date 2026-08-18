import os
import base64
import logging
from dataclasses import dataclass
from enum import Enum
import unstructured_pytesseract
from unstructured.partition.pdf import partition_pdf
logger = logging.getLogger(__name__)

unstructured_pytesseract.pytesseract.tesseract_cmd = r'D:\s\tesseract\New folder\tesseract.exe'
os.environ["PATH"] += os.pathsep + r'D:\s\poppler\New folder\poppler-26.02.0\Library\bin'


class RawChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


@dataclass
class RawChunk:

    chunk_type: RawChunkType
    page_number: int | None
    text_content: str | None = None
    image_bytes: bytes | None = None


def partition_pdf_file(file_path: str) -> list[RawChunk]:
    
    
    logger.info("Partitioning PDF: %s", file_path)

    raw_elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image"],
        extract_image_block_to_payload=True,
        chunking_strategy="by_title",
        max_characters=4000,
        combine_text_under_n_chars=1000,
        new_after_n_chars=3000,
    )

    chunks: list[RawChunk] = []

    for element in raw_elements:
        
        element_type = type(element).__name__
        
        page_number = getattr(element.metadata, "page_number", None)

        if element_type == "Table":
            
            table_html = getattr(element.metadata, "text_as_html", None)
            content = table_html if table_html else str(element)
            
            chunks.append(
                RawChunk(
                    chunk_type=RawChunkType.TABLE,
                    page_number=page_number,
                    text_content=content,
                )
            )

        elif element_type == "CompositeElement":
            
            chunks.append(
                RawChunk(
                    chunk_type=RawChunkType.TEXT,
                    page_number=page_number,
                    text_content=str(element),
                )
            )

            
            orig_elements = getattr(element.metadata, "orig_elements", None) or []
            
            for orig_el in orig_elements:
               
                if type(orig_el).__name__ == "Image":
               
                    image_b64 = getattr(orig_el.metadata, "image_base64", None)
               
                    if not image_b64:
                        continue
                    
                    try:
                        image_bytes = base64.b64decode(image_b64)
                    
                    except (ValueError, base64.binascii.Error):
                        logger.warning("Failed to decode an image block; skipping.")
                        continue
                    
                    chunks.append(
                        RawChunk(
                            chunk_type=RawChunkType.IMAGE,
                            page_number=getattr(orig_el.metadata, "page_number", page_number),
                            image_bytes=image_bytes,
                        )
                    )

        else:
           
            text = str(element).strip()
            if text:
                chunks.append(
                    RawChunk(
                        chunk_type=RawChunkType.TEXT,
                        page_number=page_number,
                        text_content=text,
                    )
                )

    logger.info(
        "Partitioned %s into %d chunks (%d text, %d table, %d image).",
        file_path,
        len(chunks),
        sum(1 for c in chunks if c.chunk_type == RawChunkType.TEXT),
        sum(1 for c in chunks if c.chunk_type == RawChunkType.TABLE),
        sum(1 for c in chunks if c.chunk_type == RawChunkType.IMAGE),
    )

    return chunks


# if __name__ == "__main__":
#     import json
#     from pathlib import Path
    
#     pdf_path = r"D:\learning\ai\proj\langchain\Multimodal RAG\uploads\AMCOR_2023Q4_EARNINGS.pdf"
#     output_dir = Path(r"D:\learning\ai\proj\langchain\Multimodal RAG\output")
#     output_dir.mkdir(exist_ok=True)
    
#     chunks = partition_pdf_file(pdf_path)
    
#     # Save metadata to JSON
#     results = []
#     for i, chunk in enumerate(chunks):
#         entry = {
#             "index": i,
#             "type": chunk.chunk_type,
#             "page_number": chunk.page_number,
#         }
#         if chunk.text_content:
#             entry["text_content"] = chunk.text_content[:500]  # truncate for readability
#         if chunk.image_bytes:
#             img_path = output_dir / f"image_{i:04d}.png"
#             img_path.write_bytes(chunk.image_bytes)
#             entry["image_path"] = str(img_path)
#         results.append(entry)
    
#     json_path = output_dir / "partition_results.json"
#     json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
#     print(f"Saved {len(chunks)} chunks to {json_path}")
