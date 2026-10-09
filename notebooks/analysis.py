####### PDF PARSE #######
import pymupdf
import re
from sentence_transformers import SentenceTransformer
import chromadb
import os 

data_folder = "data"  
all_chunks = []        

def parse_pdf(pdf_path):
    pdf = pymupdf.open(pdf_path)

    # Get paper ID from filename
    filename = pdf_path.split("/")[-1]
    paper_id_match = re.match(r"\d{4}\.\d{5}", filename)

    if paper_id_match:
        paper_id = paper_id_match.group(0)
    else:
        paper_id = filename.replace(".pdf", "")

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

            # Remove repeated dots
            text = re.sub(r"(?:\s*\.\s*){3,}", " ", text)

            # Replace line breaks with spaces
            text = re.sub(r"\s*\n\s*", " ", text)

            # Fix common missing spaces after extracted math
            text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)

            # Remove extra whitespace
            text = re.sub(r"\s+", " ", text)

            # Skip the specific table of contents
            # Checks if the page has "contents" near the beginning. Checks for "dataset details" and "supplemental results".
            # If all three conditions match, continue function skips that page instead of adding it to chunks
            lower_text = text.lower()
            if (
                "contents" in lower_text[:500]
                and "dataset details" in lower_text
                and "supplemental results" in lower_text
            ):
                continue

            pages.append(text.strip())

    pdf.close()

    return pages, paper_id, title

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

# Process every PDF in data folder
for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(data_folder, filename)

        pages, paper_id, title = parse_pdf(pdf_path)
        chunks = make_chunks(pages, paper_id, title)
        all_chunks.extend(chunks)

        print(f"Parsed {filename}: {len(chunks)} chunks")

# Count chunks across all papers
chunks = all_chunks
print("done parsing")
print(f"created {len(chunks)} chunks")

# Print first 3 chunks
# for chunk in chunks[:3]:
#     print(chunk)
#     print()



######Embed and store in ChromaDB ######

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = []

for chunk in chunks:
    texts.append(chunk["text"])

embeddings = model.encode(texts)

print("Number of embeddings:", len(embeddings))

####### Store in ChromaDB #########

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="research_papers"
)



for i, chunk in enumerate(chunks):
# Change from .add to .upsert
    collection.upsert(
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

####### RAG Retrival ##########

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


####### Questions to test #######

question1 = "I need to help my client reduce their AI cost without sacrificing the quality of the AI output. Pull recent research on how I should proceed."

question2 = "What is a quick takeaway from this article when it comes to reducing AI costs?"

question3 = "What are the business benefits of AI model routing?"


# Retrieve/display results for each question
questions = {
    "Question 1": question1,
    "Question 2": question2,
    "Question 3": question3
}

for question_name, question in questions.items():
    print(f"\n{'=' * 60}")
    print(f"{question_name}: {question}")
    print("=" * 60)

    results = retrieve_relevant_chunks(question=question, top_k=5)

    for result_number, result in enumerate(results, start=1):
        print(f"\nResult {result_number}")
        print("Title:", result["metadata"]["title"])
        print("Page:", result["metadata"]["page"])
        print("Distance:", result["distance"])
        print("Text:", result["text"][:2000])
        print("-" * 50)





    