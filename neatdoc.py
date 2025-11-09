import os
import uuid
import string
import cv2
import numpy as np
import pytesseract
import nltk
from PIL import Image
from google.cloud import storage, firestore
from google.oauth2 import service_account
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- CONFIGURATION ---
# TODO: Update these values with your GCP project details

# Path to your downloaded service account JSON file
SERVICE_ACCOUNT_FILE = 'service-account.json'  # <-- This should match the file you renamed

# Your Google Cloud project ID
GCP_PROJECT_ID = 'axiomatic-array-476616-p0'  # <-- This is from your key file

# The name of the Google Cloud Storage bucket you created
GCS_BUCKET_NAME = 'neetdocs-1nd'  

# The name for your Firestore collection (it will be created if it doesn't exist)
FIRESTORE_COLLECTION = 'neatvision_documents'

# --- DEVELOPMENT TOGGLE ---
# Set this to False to test image cleaning and text extraction locally
# Set this to True once you have your service account key and config ready
ENABLE_CLOUD_FEATURES = True
# --- END CONFIGURATION ---


def setup_nltk():
    """Downloads required NLTK data for keyword generation."""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK data not found. Downloading 'stopwords' and 'punkt'...")
        nltk.download('stopwords')
        nltk.download('punkt')
        print("NLTK data downloaded.")

def initialize_clients(service_account_path, project_id):
    """Initializes and returns the GCP Storage and Firestore clients."""
    try:
        credentials = service_account.Credentials.from_service_account_file(service_account_path)
        storage_client = storage.Client(project=project_id, credentials=credentials)
        firestore_client = firestore.Client(project=project_id, credentials=credentials)
        return storage_client, firestore_client
    except FileNotFoundError:
        print(f"Error: Service account file not found at {service_account_path}")
        print("Please update SERVICE_ACCOUNT_FILE in the script.")
        return None, None
    except Exception as e:
        print(f"Error initializing GCP clients: {e}")
        return None, None

def clean_and_enhance(image_path):
    """
    Loads a document, removes noise/stains, and enhances it for OCR.
    
    This function uses a bilateral filter to reduce noise while preserving
    edges, then uses adaptive thresholding to create a clean B&W image,
    which is excellent for OCR.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply a bilateral filter to reduce noise (stains) while keeping edges sharp
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding to get a clean, binary image
        # This is very effective for stained or unevenly lit documents
        enhanced_image = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant subtracted from the mean
        )
        
        return enhanced_image
    except Exception as e:
        print(f"Error during image processing: {e}")
        return None

def extract_text_from_image(image_data_numpy):
    """Extracts text from an OpenCV/Numpy image array using Tesseract."""
    try:
        # Convert OpenCV image (Numpy array) to PIL Image
        pil_image = Image.fromarray(image_data_numpy)
        
        # Use pytesseract to do OCR
        text = pytesseract.image_to_string(pil_image)
        return text
    except pytesseract.TesseractNotFoundError:
        print("\n" + "="*50)
        print("ERROR: Tesseract OCR Engine not found.")
        print("Please install Tesseract and ensure it's in your system's PATH.")
        print("See README.md for installation instructions.")
        print("="*50 + "\n")
        return None
    except Exception as e:
        print(f"Error during text extraction: {e}")
        return None

def generate_keywords(text_content):
    """Cleans text and generates a list of unique keywords for searching."""
    if not text_content:
        return []
        
    # Get standard English stop words
    stop_words = set(stopwords.words('english'))
    
    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text_content.translate(translator)
    
    # Tokenize the text (split into words)
    tokens = word_tokenize(text_no_punct.lower())
    
    # Filter out stop words and short tokens
    keywords = [
        word for word in tokens 
        if word.isalpha() and word not in stop_words and len(word) > 2
    ]
    
    # Return a list of unique keywords
    return list(set(keywords))

def upload_image_to_gcs(storage_client, bucket_name, image_data_numpy, destination_blob_name):
    """Uploads the enhanced image (Numpy array) to Google Cloud Storage."""
    try:
        bucket = storage_client.bucket(bucket_name)
        
        # Encode the OpenCV image (Numpy array) to a PNG byte stream
        success, buffer = cv2.imencode('.png', image_data_numpy)
        if not success:
            raise Exception("Could not encode image to PNG.")
        
        # Create a new blob and upload the byte stream
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(buffer.tobytes(), content_type='image/png')
        
        # Make the blob publicly accessible (for easier testing)
        # For production, you would use Signed URLs instead
        blob.make_public()
        
        return blob.public_url
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return None

def save_to_firestore(firestore_client, collection_name, image_url, original_text, keywords):
    """Saves the document metadata to Cloud Firestore for searching."""
    try:
        doc_ref = firestore_client.collection(collection_name).document()
        
        document_data = {
            'imageUrl': image_url,
            'originalText': original_text,
            'keywords': keywords,  # Stored as an array for 'array-contains' queries
            'timestamp': firestore.SERVER_TIMESTAMP
        }
        
        doc_ref.set(document_data)
        print(f"Successfully saved metadata to Firestore. Document ID: {doc_ref.id}")
        return doc_ref.id
    except Exception as e:
        print(f"Error saving to Firestore: {e}")
        return None

def process_document(image_path, storage_client, firestore_client, bucket_name, collection_name):
    """Runs the full processing pipeline for a single document."""
    
    print(f"1. Cleaning and enhancing image: {image_path}")
    enhanced_image = clean_and_enhance(image_path)
    if enhanced_image is None:
        return

    print("2. Extracting text from enhanced image...")
    extracted_text = extract_text_from_image(enhanced_image)
    if extracted_text is None:
        return
    print(f"   - Extracted text (snippet): '{extracted_text[:75].strip()}...'")

    print("3. Generating search keywords...")
    keywords = generate_keywords(extracted_text)
    print(f"   - Found {len(keywords)} unique keywords.")

    # Create a unique file name for the GCS blob
    destination_blob_name = f"processed/{uuid.uuid4()}.png"

    print(f"4. Uploading enhanced image to GCS at: {destination_blob_name}")
    image_url = upload_image_to_gcs(
        storage_client, 
        bucket_name, 
        enhanced_image, 
        destination_blob_name
    )
    if image_url is None:
        return
    print(f"   - Image URL: {image_url}")

    print("5. Saving text and metadata to Firestore...")
    save_to_firestore(
        firestore_client, 
        collection_name, 
        image_url, 
        extracted_text, 
        keywords
    )
    
    print("\n--- Document Processing Complete ---")

def search_by_text(firestore_client, collection_name, search_term):
    """Searches Firestore for documents containing a specific keyword."""
    
    # Clean the search term to match a single keyword
    cleaned_term = search_term.lower().strip(string.punctuation)
    if not cleaned_term:
        print("Invalid search term.")
        return []

    print(f"Querying Firestore for documents where 'keywords' array contains '{cleaned_term}'...")
    
    try:
        collection_ref = firestore_client.collection(collection_name)
        # Use 'array-contains' to efficiently query the keywords list
        query = collection_ref.where('keywords', 'array_contains', cleaned_term)
        
        results = []
        for doc in query.stream():
            results.append(doc.to_dict())
            
        return results
    except Exception as e:
        print(f"Error during Firestore search: {e}")
        return []


if __name__ == "__main__":
    # 1. Download NLTK data (one-time setup)
    setup_nltk()

    # --- Define your test image here ---
    # TODO: Change this to the path of your test image
    test_image = 'my_document.png' 
    
    if not os.path.exists(test_image):
        print(f"Test image not found: {test_image}")
        print("Please download a sample document image and save it as 'my_document.png' or update the path.")
    
    # --- Run in Cloud Mode or Local Mode ---
    elif ENABLE_CLOUD_FEATURES:
        # 2. Initialize GCP Clients
        print(f"Cloud mode enabled. Initializing clients for project '{GCP_PROJECT_ID}'...")
        storage_client, db = initialize_clients(SERVICE_ACCOUNT_FILE, GCP_PROJECT_ID)
        
        if storage_client and db:
            print("Clients initialized successfully.")
            
            # 3. --- Process a Document (Cloud) ---
            process_document(
                test_image, 
                storage_client, 
                db, 
                GCS_BUCKET_NAME, 
                FIRESTORE_COLLECTION
            )

            # 4. --- Search for a Document (Cloud) ---
            # TODO: Change 'python' to a word you expect to find in your document
            search_word = 'python' 
            
            print("\n" + "="*50)
            print(f"Searching for documents containing the word '{search_word}'...")
            
            search_results = search_by_text(db, FIRESTORE_COLLECTION, search_word)
            
            if search_results:
                print(f"Found {len(search_results)} matching document(s):")
                for i, result in enumerate(search_results):
                    print(f"\n--- Result {i+1} ---")
                    print(f"  Image URL: {result.get('imageUrl')}")
                    print(f"  Text (snippet): {result.get('originalText', '')[:150].strip()}...")
            else:
                print(f"No documents found containing the word '{search_word}'.")
            print("="*50)
        else:
            print("Could not initialize cloud clients. Please check your configuration.")

    else:
        # --- Run in Local-Only Mode ---
        print("\nCloud features are DISABLED. Running in local-only test mode.")
        print("This will test cleaning, OCR, and keyword generation.")
        
        print(f"1. Cleaning and enhancing image: {test_image}")
        enhanced_image = clean_and_enhance(test_image)
        
        if enhanced_image is not None:
            print("2. Extracting text from enhanced image...")
            extracted_text = extract_text_from_image(enhanced_image)
            
            if extracted_text is not None:
                print(f"   - Extracted text (snippet): '{extracted_text[:75].strip()}...'")
                
                print("3. Generating search keywords...")
                keywords = generate_keywords(extracted_text)
                print(f"   - Found {len(keywords)} unique keywords.")

                # 4. Save results locally
                output_image_path = "LOCAL_enhanced_document.png"
                output_text_path = "LOCAL_extracted_results.txt"
                
                try:
                    cv2.imwrite(output_image_path, enhanced_image)
                    print(f"4. Successfully saved enhanced image to: {output_image_path}")
                except Exception as e:
                    print(f"Error saving enhanced image: {e}")
                
                try:
                    with open(output_text_path, 'w', encoding='utf-8') as f:
                        f.write("--- EXTRACTED TEXT ---\n")
                        f.write(extracted_text)
                        f.write("\n\n--- KEYWORDS ---\n")
                        f.write(", ".join(keywords))
                    print(f"5. Successfully saved text and keywords to: {output_text_path}")
                except Exception as e:
                    print(f"Error saving text file: {e}")

        print("\n--- Local Processing Complete ---")

