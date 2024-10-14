import asyncio
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_to_text_converter import convert_to_text

async def test_convert_to_text(url):
    try:
        text = await convert_to_text(url)
        print(f"Converted text (first 500 characters):\n{text[:500]}")
    except Exception as e:
        print(f"Error converting PDF: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_pdf_to_text_converter.py <pdf_url>")
        sys.exit(1)

    pdf_url = sys.argv[1]
    asyncio.run(test_convert_to_text(pdf_url))
