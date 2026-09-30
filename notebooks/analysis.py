####### PDF PARSE #######
import pymupdf
import re

pdf_path = "data/2608.20316v1_Pandora's AI Model Routing Box Efficient Allocation with Costly Value Estimation.pdf"
pdf = pymupdf.open(pdf_path)

# Get paper ID from filename
filename = pdf_path.split("/")[-1]
paper_id = re.match(r"\d{4}\.\d{5}", filename).group(0)

# Get title from the first page
first_page_lines = [
    line.strip()
    for line in pdf[0].get_text().split("\n")
    if line.strip()
]

# Title spans two lines
title = " ".join(first_page_lines[1:3]).strip()

# Store cleaned text for each page
pages = []

for page in pdf:
    text = page.get_text()

    if text:
        # Manually remove asterisks from author names only
        text = text.replace("Adam Fisch∗", "Adam Fisch")
        text = text.replace("Jacob Eisenstein∗", "Jacob Eisenstein")

        # Fix words split across lines
        text = re.sub(r"-\s*\n\s*", "", text)

        # Remove repeated dots (from the table of contents)
        text = re.sub(r"(?:\s*\.\s*){3,}", " ", text)

        # Replace line breaks with spaces
        text = re.sub(r"\s*\n\s*", " ", text)

        # Fix common missing spaces after extracted math
        text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)

        pages.append(text.strip())

pdf.close()


####### TEXT CHUNKS (currently 1 chunk per page) #######

def make_chunks(pages, paper_id, title):
    chunks = []

    for page_number, text in enumerate(pages, start=1):
        chunks.append({
            "chunk_id": f"{paper_id}_{page_number:03d}",
            "paper_id": paper_id,
            "title": title,
            "page": page_number,
            "text": text
        })

    return chunks

chunks = make_chunks(pages, paper_id, title)
print("done parsing")
print(f"created {len(chunks)} chunks")

# Print first 3 chunks
for chunk in chunks[:3]:
    print(chunk)
    print()