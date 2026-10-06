####### PDF PARSE #######
import pymupdf
import re
from sentence_transformers import SentenceTransformer
import chromadb

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
# for chunk in chunks[:3]:
#     print(chunk)
#     print()





# sample_chunks = [
#     {
#         "chunk_id": "test_001",
#         "paper_id": "2608.20318",
#         "title": "AI4AI-Bench",
#         "page": 1,
#         "text": "This paper evaluates LLM agents using benchmark tasks."
#     },

#     {
#         "chunk_id": "test_002",
#         "paper_id": "2608.20318",
#         "title": "AI4AI-Bench",
#         "page": 2,
#         "text": "The researchers compare different methods for evaluating AI agents."
#     },

#     {
#         "chunk_id": "test_003",
#         "paper_id": "paper_002",
#         "title": "Agentic AI Research",
#         "page": 1,
#         "text": "Agentic AI can assist researchers with collecting and organizing data."
#     }
# ]


model = SentenceTransformer("all-MiniLM-L6-v2")


texts = []

for chunk in chunks:
    texts.append(chunk["text"])



embeddings = model.encode(texts)

print("Number of embeddings:", len(embeddings))



client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="research_papers"
)



for i, chunk in enumerate(chunks):

    collection.add(
        ids=[chunk["chunk_id"]],

        embeddings=[
            embeddings[i].tolist()
        ],

        documents=[
            chunk["text"]
        ],

        metadatas=[
            {
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "page": chunk["page"]
            }
        ]
    )


print("Chunks successfully stored!")


def retrieve_relevant_chunks(question, top_k = 5) :
    question_embedding = model.encode(question).tolist()

    results = collection.query(query_embeddings = [question_embedding],
    n_results = top_k
    )

    retrieved_chunks = []

    for index in range(len(results["documents"][0])):
        retrieved_chunk = {
            "text": results["documents"][0][index],
            "metadata": results["metadatas"][0][index],
            "distance": results["distances"][0][index],
        }
        retrieved_chunks.append(retrieved_chunk)

    return retrieved_chunks


question = "I need to help my client reduce their AI cost without sacrificing the quality of the AI output. Pull recent research on how I should proceed ?"

results = retrieve_relevant_chunks(question = question, top_k = 5)

for result_number, result in enumerate(results, start = 1):

    print(f"\nResult {result_number}")
    print("Title: ", result["metadata"]["title"])
    print("Page: ", result["metadata"]["page"])
    print("Distance:", result["distance"])
    print("Text:", result["text"][:2000])
    print("-" * 50)

    