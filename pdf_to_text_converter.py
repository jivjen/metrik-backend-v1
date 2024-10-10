import requests
import os
import fitz  

def convert_to_text(file_url: str) -> str:
    def jina_ai_conversion():
        try:
            url = f'https://r.jina.ai/{file_url}'
            headers = {
                "Authorization": "Bearer jina_cdfde91597854ce89ef3daed22947239autBdM5UrHeOgwRczhd1JYzs51OH"
            }
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                print(f"Successfully converted {file_url} to text using Jina AI")
                print(f"READ PDF + {response.text[:100]}")
                return response.text
            else:
                print(f"Error converting {file_url} to text using Jina AI. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error converting {file_url} to text using Jina AI: {e}")
            return None

    def mupdf_conversion():
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            # Download the file with user-agent header
            response = requests.get(file_url, headers=headers)
            if response.status_code != 200:
                print(f"Error downloading file from {file_url}. Status code: {response.status_code}")
                return ""

            # Save the file temporarily
            file_name = os.path.basename(file_url)
            with open(file_name, 'wb') as f:
                f.write(response.content)

            # Convert PDF to text using PyMuPDF
            doc = fitz.open(file_name)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Remove the temporary file
            os.remove(file_name)

            print(f"Successfully converted {file_url} to text using PyMuPDF")
            print(f"READ PDF MuPdf + {text[:100]}")
            return text
        except Exception as e:
            print(f"Error converting {file_url} to text using PyMuPDF: {e}")
            return ""

    # Try Jina AI first, then fall back to PyMuPDF
    text = jina_ai_conversion()
    print(f"length of text = {len(text)}")
    if not text or len(text)<3000:
        print(f"Jina AI text = {text}")
        print("Jina AI conversion failed. Trying PyMuPDF...")
        text = mupdf_conversion()

    return text