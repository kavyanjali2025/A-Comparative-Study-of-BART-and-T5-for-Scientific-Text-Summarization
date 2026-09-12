# pypdf is a Python library for working with PDF files.
from pypdf import PdfReader

print("pypdf is working!")

reader = PdfReader("paper.pdf") # now reader represents the PDF document

print(f"The PDF has {len(reader.pages)} pages")

# extracting the ENTIRE paper
text = ""

for page in reader.pages:
    text += page.extract_text()

print(text)

