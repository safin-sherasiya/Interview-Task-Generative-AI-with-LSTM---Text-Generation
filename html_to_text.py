from bs4 import BeautifulSoup

# Load the HTML file
with open("pg100-images.html", "r", encoding="utf-8") as f:
    html = f.read()

# Parse the HTML content
soup = BeautifulSoup(html, "html.parser")

# Extract text from HTML while removing tags
text = soup.get_text()

# Save the extracted text to a file
with open("shakespeare.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Text extraction completed successfully.")
