from pypdf import PdfReader
from tqdm import tqdm


def load_pdf(file_path: str) -> list[dict]:
    reader = PdfReader(file_path)
    pages = []

    print("Extracting text from PDF...")

    for i, page in enumerate(tqdm(reader.pages, desc="PDF Pages")):
        text = page.extract_text()
        if text:
            pages.append({
                "page": i + 1,
                "text": text
            })

    return pages