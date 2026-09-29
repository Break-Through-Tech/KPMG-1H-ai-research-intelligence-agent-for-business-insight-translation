# Library that does PDF parsing
import PyPDF2
pdf_path = "../data/2608.20316v1_Pandora's AI Model Routing Box Efficient Allocation with Costly Value Estimation.pdf"
pdf_reader = PyPDF2.PdfReader(pdf_path)

# From module 7.5 in MLF
file_content = ""
for page in pdf_reader.pages:
    text = page.extract_text()
    if text:
        file_content += text + "\n"

if file_content:
    print("done parsing")
    print(file_content)
else:
    print("parse didnt work")


# the above works: it parses and prints a specific doc


# STEP 2: CONNECT TO ollama?
#ai_connection =


