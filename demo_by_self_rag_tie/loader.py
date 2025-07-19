from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.image import SimpleImageReader
from typing import List

def load_documents(file_paths: List[str]):
    pdf_reader = PyMuPDFReader()
    image_reader = SimpleImageReader()

    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            docs.extend(pdf_reader.load(file_path=path))
        elif path.lower().endswith((".jpg", ".jpeg", ".png")):
            docs.extend(image_reader.load_data(path))
    return docs
