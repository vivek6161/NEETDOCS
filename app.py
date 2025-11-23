import nltk

try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")
# app.py
import os
import json
import tempfile

if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
    creds_dict = json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(json.dumps(creds_dict).encode())
        tmp.flush()
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp.name

import os
import uuid
import string
import cv2
import numpy as np
import pytesseract
import nltk
import logging
from PIL import Image
# --- CLOUD IMPORTS ---
from google.cloud import storage, firestore
from google.oauth2 import service_account
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import (
    Flask,
    request,
    render_template_string,
    send_file,
    after_this_request,
    redirect,
    url_for,
    make_response,
    session,
    flash
)
from datetime import timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# --- 1. CONFIGURATION ---

# Your Google Cloud project ID
GCP_PROJECT_ID = 'axiomatic-array-476616-p0'

# The name of the Google Cloud Storage bucket
GCS_BUCKET_NAME = 'neetdocs-1nd'

# The name for your Firestore collection for documents
FIRESTORE_COLLECTION = 'neatvision_documents'

# The name for your Firestore collection for users (auth)
FIRESTORE_USERS_COLLECTION = 'neatvision_users'

# Path for temporary file storage (Local)
UPLOAD_FOLDER = 'temp_uploads'

# Path to your downloaded service account JSON file
SERVICE_ACCOUNT_FILE = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

ENABLE_CLOUD_FEATURES = True

# SECRET KEY for sessions (replace with secure key or env var in production)
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace_this_with_a_secure_random_key")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- END CONFIGURATION ---

# --- 2. INITIALIZATION & SETUP ---

logging.basicConfig(level=logging.INFO)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

storage_client = None
db = None

def setup_nltk():
    """Downloads required NLTK data for keyword generation."""
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logging.info("NLTK data not found. Downloading 'stopwords' and 'punkt'...")
        nltk.download('stopwords')
        nltk.download('punkt')
        logging.info("NLTK data downloaded.")

def initialize_clients(service_account_path, project_id):
    """Initializes GCP clients using service account file."""
    global storage_client, db
    try:
        credentials = service_account.Credentials.from_service_account_file(service_account_path)
        storage_client = storage.Client(project=project_id, credentials=credentials)
        db = firestore.Client(project=project_id, credentials=credentials)
        logging.info(f"GCP Clients initialized for project: {project_id}")
        return True
    except FileNotFoundError:
        logging.error(f"Error: Service account file not found at {service_account_path}")
        return False
    except Exception as e:
        logging.error(f"Error initializing GCP clients: {e}")
        return False

# Run setup on startup
setup_nltk()
if ENABLE_CLOUD_FEATURES:
    GCP_CLIENTS_READY = initialize_clients(SERVICE_ACCOUNT_FILE, GCP_PROJECT_ID)
else:
    GCP_CLIENTS_READY = False

# --- 3. AUTH HELPERS (LOGIN) ---

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return wrapper

# --- 4. CORE PROCESSING FUNCTIONS ---

def allowed_file(filename):
    """Checks if the file has an allowed image extension."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_and_enhance(image_path):
    """
    Loads and enhances the document image.
    FIX: Adjusted parameters for Adaptive Thresholding to prevent deleting fine text.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Could not read image from {image_path}")
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Keep Median Blur to remove any minor digital noise
        denoised = cv2.medianBlur(gray, 3)

        # CRITICAL FIX: Constant C reduced from 10 to 2 to preserve thin lines (like on certificates)
        enhanced_image = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 2  # Constant C reduced from 10 to 2
        )

        kernel = np.ones((2,2), np.uint8)
        final_image = cv2.morphologyEx(enhanced_image, cv2.MORPH_OPEN, kernel)
        return final_image
    except Exception as e:
        logging.error(f"Error during image processing: {e}")
        return None

def extract_text_from_image(image_data_numpy):
    """Extracts text from an enhanced image using Tesseract."""
    try:
        pil_image = Image.fromarray(image_data_numpy)
        text = pytesseract.image_to_string(pil_image)
        return text
    except pytesseract.TesseractNotFoundError:
        logging.error("Tesseract OCR Engine not found. Please install it.")
        return None
    except Exception as e:
        logging.error(f"Error during text extraction: {e}")
        return None

def generate_keywords(text_content):
    """Cleans text and generates a list of unique keywords for searching."""
    if not text_content: return []
    stop_words = set(stopwords.words('english'))
    translator = str.maketrans('', '', string.punctuation)
    text_no_punct = text_content.translate(translator)
    tokens = word_tokenize(text_no_punct.lower())
    keywords = [
        word for word in tokens
        if word.isalpha() and word not in stop_words and len(word) > 2
    ]
    return list(set(keywords))

def upload_image_to_gcs(storage_client, bucket_name, image_data_numpy, destination_blob_name):
    """Uploads the enhanced image to GCS and returns its blob name (path)."""
    try:
        bucket = storage_client.bucket(bucket_name)
        success, buffer = cv2.imencode('.png', image_data_numpy)
        if not success:
            raise Exception("Could not encode image to PNG.")

        blob = bucket.blob(destination_blob_name)
        blob.upload_from_string(buffer.tobytes(), content_type='image/png')

        return blob.name  # We save the internal path (blob name)
    except Exception as e:
        logging.error(f"Error uploading to GCS: {e}")
        return None

def generate_signed_url(storage_client, blob_name, expiration_time=3600):
    """Generates a temporary, secure Signed URL for private GCS objects."""
    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)

        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=expiration_time),
            method="GET",
        )
        return url
    except Exception as e:
        logging.error(f"Error generating signed URL for {blob_name}: {e}")
        return ""

def save_to_firestore(firestore_client, collection_name, image_blob_name, original_text, keywords):
    """Saves the document metadata to Cloud Firestore."""
    try:
        doc_ref = firestore_client.collection(collection_name).document()
        document_data = {
            # We save the internal GCS blob name, not the URL
            'imageBlobName': image_blob_name,
            'originalText': original_text,
            'keywords': keywords,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'docId': doc_ref.id
        }
        doc_ref.set(document_data)
        logging.info(f"Saved metadata to Firestore. Document ID: {doc_ref.id}")
        return doc_ref.id
    except Exception as e:
        logging.error(f"Error saving to Firestore: {e}")
        return None

def search_by_text(firestore_client, collection_name, search_term):
    """Searches Firestore for documents containing a specific keyword."""

    cleaned_term = search_term.lower().strip(string.punctuation)
    if not cleaned_term:
        return []

    logging.info(f"Querying Firestore for documents containing '{cleaned_term}'...")

    try:
        collection_ref = firestore_client.collection(collection_name)
        query = collection_ref.where(filter=firestore.FieldFilter('keywords', 'array_contains', cleaned_term))

        results = []
        for doc in query.stream():
            doc_data = doc.to_dict()

            # CRITICAL: Generate Signed URL for image display
            if 'imageBlobName' in doc_data and storage_client is not None:
                doc_data['imageUrl'] = generate_signed_url(
                    storage_client,
                    doc_data['imageBlobName']
                )
            results.append(doc_data)

        logging.info(f"Found {len(results)} matching document(s).")
        return results
    except Exception as e:
        logging.error(f"Error during Firestore search: {e}")
        return []

# --- 5. HTML TEMPLATE (Your original dashboard UI retained) ---

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>📄 ProScan Document Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #f4f6f9;
            --card-bg: #ffffff;
            --primary-blue: #007bff;
            --success-green: #28a745;
            --warning-orange: #ffc107;
            --text-dark: #343a40;
            --text-light: #6c757d;
            --border-color: #e9ecef;
            --shadow-subtle: 0 0.5rem 1rem rgba(0, 0, 0, 0.05);
        }
        body {
            font-family: 'Roboto', sans-serif;
            background-color: var(--bg-color);
            min-height: 100vh;
            margin: 0;
            padding: 40px 20px;
            color: var(--text-dark);
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .main-container {
            max-width: 1000px;
            width: 100%;
        }
        header {
            text-align: center;
            margin-bottom: 20px;
        }
        header h1 {
            font-size: 2.5rem;
            color: var(--primary-blue);
            font-weight: 700;
            margin-bottom: 5px;
        }
        .subtitle {
            color: var(--text-light);
            font-size: 1.1rem;
        }

        .topbar {
            display:flex;
            justify-content:space-between;
            align-items:center;
            margin-bottom: 20px;
        }
        .topbar .user {
            color: var(--text-light);
            font-size: 0.95rem;
        }
        .topbar a { color: var(--primary-blue); text-decoration:none; margin-left:10px; }

        /* --- MAIN GRID LAYOUT --- */
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
            margin-bottom: 40px;
        }
        @media (min-width: 768px) {
            .dashboard-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
        /* --- CARD STYLES --- */
        .card {
            background-color: var(--card-bg);
            padding: 30px;
            border-radius: 8px;
            box-shadow: var(--shadow-subtle);
            border: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
        }
        .card h2 {
            font-size: 1.5rem;
            font-weight: 500;
            color: var(--text-dark);
            margin-bottom: 25px;
            padding-bottom: 5px;
            border-bottom: 2px solid var(--border-color);
        }
        /* --- INPUT & BUTTON STYLES --- */
        .upload-field-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
            padding: 20px 15px;
            border: 2px dashed var(--primary-blue);
            border-radius: 6px;
            cursor: pointer;
            text-align: center;
            transition: all 0.2s ease;
            margin-bottom: 20px;
        }
        .upload-field-group:hover {
            background-color: #f0f8ff;
        }

        input[type="file"], input[type="text"] {
            display: none;
        }
        .file-label strong {
            display: block;
            color: var(--text-dark);
            font-size: 1.1em;
            margin-top: 5px;
        }
        .status-text {
            font-size: 0.9em;
            display: block;
            margin-top: 5px;
            color: var(--text-light);
        }

        .search-form input[type="text"] {
            display: block;
            width: 100%;
            padding: 10px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 1rem;
            margin-bottom: 20px;
        }

        button {
            width: 100%;
            padding: 12px 20px;
            border: none;
            border-radius: 6px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
        }
        .upload-button { background-color: var(--success-green); }
        .upload-button:hover { background-color: #1e7e34; }
        .search-button { background-color: var(--primary-blue); }
        .search-button:hover { background-color: #0056b3; }

        /* Loading Styles */
        button:disabled { background-color: var(--text-light); cursor: wait; }
        .spinner { border: 3px solid rgba(255, 255, 255, 0.3); border-top: 3px solid white; border-radius: 50%; width: 14px; height: 14px; animation: spin 0.8s linear infinite; margin-right: 8px; display: none; }
        .loading .spinner { display: block; }
        .loading .button-text { visibility: hidden; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* --- RESULTS SECTION --- */
        .results-list {
            margin-top: 30px;
            display: grid;
            gap: 20px;
            grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
        }
        .result-card {
            background-color: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid var(--primary-blue);
            box-shadow: var(--shadow-subtle);
            display: flex;
            gap: 15px;
        }
        .result-card img {
            width: 150px;
            height: 200px;
            object-fit: cover;
            border-radius: 4px;
            flex-shrink: 0;
            border: 1px solid var(--border-color);
        }
        .metadata {
            text-align: left;
            flex-grow: 1;
        }
        .metadata h3 {
            font-size: 1rem;
            color: var(--primary-blue);
            margin: 0 0 5px 0;
        }
        .keywords-list {
            margin-top: 10px;
            padding-top: 5px;
            border-top: 1px dashed var(--border-color);
        }
        .keywords-list span {
            display: inline-block;
            background: #e0f7fa;
            color: #00796b;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.75rem;
            margin-right: 5px;
            margin-bottom: 5px;
        }
        .download-btn {
            background-color: var(--primary-blue);
            color: white;
            padding: 6px 10px;
            border-radius: 4px;
            text-decoration: none;
            font-size: 0.9rem;
            margin-top: 10px;
            display: inline-block;
        }
        .text-snippet {
            font-style: italic;
            font-size: 0.9rem;
            color: var(--text-light);
            margin-bottom: 10px;
        }
        .auth-link { text-align: right; margin-bottom: 8px; }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="topbar">
            <div>
                <header>
                    <h1>ProScan AI Document Indexer</h1>
                    <p class="subtitle">Secure, fast indexing and full-text search powered by Computer Vision and Google Cloud.</p>
                </header>
            </div>
            <div class="user">
                {% if current_user %}
                    Logged in as: <strong>{{ current_user }}</strong>
                    <a href="{{ url_for('logout') }}">Logout</a>
                {% else %}
                    <a href="{{ url_for('login') }}">Login</a> | <a href="{{ url_for('register') }}">Register</a>
                {% endif %}
            </div>
        </div>

        {% if gcp_error %}
        <div class="card" style="border-left: 5px solid red; color: red; margin-bottom: 20px;">
            🚨 CLOUD ERROR: GCP services are not fully initialized. Check your `app.py` configuration and service account roles.
        </div>
        {% endif %}
        
        <div class="dashboard-grid">
            <!-- Left Side: Document Upload -->
            <div class="card">
                <h2>1. Index New Document</h2>
                <form action="{{ url_for('process_document_route') }}" method="POST" enctype="multipart/form-data" id="uploadForm">
                    <input type="file" name="file" accept="image/png, image/jpeg, image/jpg" id="fileInput" required>
                    
                    <label for="fileInput" class="upload-field-group">
                        <span style="font-size: 2rem; color: var(--primary-blue);">📄</span>
                        <strong>Click or Drag to Upload</strong>
                        <span id="fileName" class="status-text" style="color: var(--text-light);">Maximum recommended size: 5MB</span>
                    </label>

                    <button type="submit" id="submitButton" class="upload-button">
                        <div class="spinner"></div>
                        <span class="button-text">🚀 Process and Index Document</span>
                    </button>
                </form>
            </div>

            <!-- Right Side: Document Search -->
            <div class="card">
                <h2>2. Search Index</h2>
                <form action="{{ url_for('search_documents_route') }}" method="GET" id="searchForm" class="search-form">
                    <input type="text" name="query" placeholder="Search by document content keyword..." required value="{{ search_query or '' }}">
                    <button type="submit" id="searchButton" class="search-button">
                        <div class="spinner"></div>
                        <span class="button-text">🔍 Search Documents</span>
                    </button>
                </form>
            </div>
        </div>
    
        <!-- Search Results Section -->
        {% if search_query is not none %}
        <div class="results-container" style="width: 100%;">
            <h2 style="color: var(--text-dark); margin-top: 0; font-size: 1.5rem;">
                Search Results for: "<span style="color: var(--primary-blue);">{{ search_query }}</span>" 
                ({{ results | length }} Documents Found)
            </h2>
            <div class="results-list">
                {% for doc in results %}
                <div class="result-card">
                    <img src="{{ doc.imageUrl }}" alt="Processed Document Thumbnail" onerror="this.onerror=null;this.src='https://placehold.co/150x200/cccccc/333333?text=Image%20Error';" >
                    <div class="metadata">
                        <h3>Document ID: {{ doc.docId }}</h3>
                        <p class="text-snippet">{{ doc.originalText | truncate(150) }}</p>
                        <div class="keywords-list">
                            {% for keyword in doc.keywords[:10] %}
                            <span>{{ keyword }}</span>
                            {% endfor %}
                        </div>
                        <a href="{{ url_for('download_blob', blob_name=doc.imageBlobName, doc_id=doc.docId) }}" class="download-btn">
                            📥 Download Cleaned Image
                        </a>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>
    <script>
        // --- Shared Function for Loading State ---
        function setLoading(buttonId, isLoading, text) {
            const button = document.getElementById(buttonId);
            if (!button) return;
            button.disabled = isLoading;
            button.classList.toggle('loading', isLoading);
            button.querySelector('.button-text').textContent = text;
        }

        // --- Upload Form Logic ---
        document.getElementById('fileInput').addEventListener('change', function() {
            const fileNameDisplay = document.getElementById('fileName');
            if (this.files.length > 0) {
                fileNameDisplay.textContent = 'Selected: ' + this.files[0].name;
                fileNameDisplay.style.color = '#28a745';
            } else {
                fileNameDisplay.textContent = 'Maximum recommended size: 5MB';
                fileNameDisplay.style.color = '#6c757d';
            }
        });

        document.getElementById('uploadForm').addEventListener('submit', function(e) {
            const fileInput = document.getElementById('fileInput');
            if (fileInput.files.length === 0) {
                alert("Please select an image file to process.");
                e.preventDefault();
                return;
            }
            setLoading('submitButton', true, 'Processing... Uploading to Cloud.');
        });
        
        // --- Search Form Logic ---
        document.getElementById('searchForm').addEventListener('submit', function(e) {
            const queryInput = document.querySelector('.search-form input[name="query"]');
            if (!queryInput.value.trim()) {
                alert("Please enter a keyword to search.");
                e.preventDefault();
                return;
            }
            setLoading('searchButton', true, 'Searching...');
        });
        
        // --- Truncate Filter for Jinja compatibility (simple implementation) ---
        String.prototype.truncate = function(n) {
            return (this.length > n) ? this.substr(0, n - 1) + '...' : this;
        };
        
    </script>
</body>
</html>
"""

# --- 6. FLASK TEMPLATE HELPERS ---

def truncate_filter(s, length=250):
    # This filter function will handle the truncation of text for the UI display
    if s is None:
        return ""
    return s[:length] + '...' if len(s) > length else s

# --- 7. AUTH ROUTES (REGISTER / LOGIN / LOGOUT) ---

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")

        if not email or not password:
            return "Email and password required.", 400

        if ENABLE_CLOUD_FEATURES and db is not None:
            user_ref = db.collection(FIRESTORE_USERS_COLLECTION).document(email)
            if user_ref.get().exists:
                return "User already exists. Please login.", 400

            user_ref.set({
                "password_hash": generate_password_hash(password),
                "created_at": firestore.SERVER_TIMESTAMP
            })
            return redirect(url_for("login"))
        else:
            # Local fallback (not using Firestore)
            return "Cloud features not enabled or DB not initialized.", 500

    # Simple register form that fits into your app flow
    return """
    <div style="max-width:420px;margin:80px auto;font-family:Roboto, sans-serif;">
        <h2>Create Account</h2>
        <form method="POST">
            <label>Email</label><br>
            <input name="email" type="email" required style="width:100%;padding:8px;margin:6px 0;"><br>
            <label>Password</label><br>
            <input name="password" type="password" required style="width:100%;padding:8px;margin:6px 0;"><br>
            <button type="submit" style="padding:10px 16px;background:#007bff;color:white;border:none;border-radius:6px;">Create Account</button>
        </form>
        <p style="margin-top:12px;">Already have an account? <a href="/login">Login</a></p>
    </div>
    """

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").lower().strip()
        password = request.form.get("password", "")

        if not email or not password:
            return "Email and password required.", 400

        if ENABLE_CLOUD_FEATURES and db is not None:
            user_doc = db.collection(FIRESTORE_USERS_COLLECTION).document(email).get()
            if not user_doc.exists:
                return "Invalid email or password.", 400
            user_data = user_doc.to_dict()
            stored_hash = user_data.get("password_hash", "")
            if not check_password_hash(stored_hash, password):
                return "Invalid email or password.", 400

            session["user"] = email
            # optionally set session lifetime
            session.permanent = True
            app.permanent_session_lifetime = timedelta(days=7)

            next_url = request.args.get("next")
            return redirect(next_url or url_for("index"))
        else:
            return "Cloud features not enabled or DB not initialized.", 500

    return """
    <div style="max-width:420px;margin:80px auto;font-family:Roboto, sans-serif;">
        <h2>Login</h2>
        <form method="POST">
            <label>Email</label><br>
            <input name="email" type="email" required style="width:100%;padding:8px;margin:6px 0;"><br>
            <label>Password</label><br>
            <input name="password" type="password" required style="width:100%;padding:8px;margin:6px 0;"><br>
            <button type="submit" style="padding:10px 16px;background:#007bff;color:white;border:none;border-radius:6px;">Login</button>
        </form>
        <p style="margin-top:12px;">Don't have an account? <a href="/register">Register</a></p>
    </div>
    """

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# --- 8. FLASK ROUTES (INDEX / PROCESS / SEARCH / DOWNLOAD) ---

@app.route("/", methods=['GET'])
@login_required
def index():
    """Renders the main page with upload and search forms."""
    return render_template_string(
        INDEX_TEMPLATE,
        gcp_error=not GCP_CLIENTS_READY,
        search_query=None,
        results=None,
        current_user=session.get("user")
    )

@app.route("/process", methods=['POST'])
@login_required
def process_document_route():
    """Handles file upload, processing, and saves data to GCS/Firestore."""
    if not GCP_CLIENTS_READY:
        return redirect(url_for('index', gcp_error=True))

    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return "Invalid file or no selected file", 400

    # Local Path Setup
    original_extension = file.filename.rsplit('.', 1)[1].lower()
    original_filename = f"{uuid.uuid4()}.{original_extension}"
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    file.save(original_path)

    # --- Processing Pipeline ---
    enhanced_image_data = clean_and_enhance(original_path)
    if enhanced_image_data is None:
        return "Image processing failed.", 500

    extracted_text = extract_text_from_image(enhanced_image_data)
    if extracted_text is None:
        return "Text extraction failed. Tesseract might not be installed.", 500

    keywords = generate_keywords(extracted_text)

    # GCS Path Setup
    destination_blob_name = f"processed/{uuid.uuid4()}.png"
    # IMPORTANT: We upload the blob name, not the URL, since we use Signed URLs later
    image_blob_name = upload_image_to_gcs(
        storage_client, GCS_BUCKET_NAME, enhanced_image_data, destination_blob_name
    )

    if image_blob_name is None:
        return "GCS upload failed. Check service account permissions (Storage Object Admin).", 500

    # Firestore Save
    # Save the blob name, not the URL
    doc_id = save_to_firestore(
        db, FIRESTORE_COLLECTION, image_blob_name, extracted_text, keywords
    )

    # Cleanup local file
    @after_this_request
    def cleanup(response):
        try:
            os.remove(original_path)
        except Exception as e:
            logging.warning(f"Error cleaning up file {original_path}: {e}")
        return response

    # Redirect to search with a keyword
    return redirect(url_for('search_documents_route', query=keywords[0] if keywords else ''))

@app.route("/download_blob/<path:blob_name>/<doc_id>", methods=['GET'])
@login_required
def download_blob(blob_name, doc_id):
    """
    Retrieves the file content from GCS using the blob_name and serves it for download.
    """
    if not GCP_CLIENTS_READY:
        return "Cloud clients not initialized.", 500

    logging.info(f"Attempting to download blob: {blob_name}")

    try:
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(blob_name)

        # Download the file contents into memory
        file_bytes = blob.download_as_bytes()

        # Create a Flask response with the file bytes
        response = make_response(file_bytes)

        # Set headers for download
        response.headers["Content-Disposition"] = f"attachment; filename=cleaned_document_{doc_id}.png"
        response.headers["Content-Type"] = "image/png"

        return response

    except Exception as e:
        logging.error(f"Error during file download for {blob_name}: {e}")
        return "Error downloading file from storage. Check service account permissions.", 500

@app.route("/search", methods=['GET'])
@login_required
def search_documents_route():
    """Handles the search query and displays results."""
    if not GCP_CLIENTS_READY:
        return render_template_string(INDEX_TEMPLATE, gcp_error=True, search_query=None, results=[], current_user=session.get("user"))

    search_query = request.args.get('query', '').strip()

    results = []
    if search_query:
        results = search_by_text(db, FIRESTORE_COLLECTION, search_query)

    # Render the main template with the search results and the active query
    return render_template_string(
        INDEX_TEMPLATE,
        gcp_error=False,
        search_query=search_query,
        results=results,
        truncate=truncate_filter,
        current_user=session.get("user")
    )

# --- 9. RUN THE APP ---

if __name__ == "__main__":
    logging.info("Starting the web server...")
    app.run(debug=True, port=int(os.environ.get("PORT", 5000))) 

