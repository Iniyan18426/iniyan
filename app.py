import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import uuid
from flask import Flask, render_template, request, jsonify, send_from_directory, session
from PIL import Image
import shutil
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as text_cosine_similarity
from flask_cors import CORS
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'your-secret-key-here'  # Change this to a secure random key

# Configure paths
UPLOAD_FOLDER = 'static/uploads'
DATASET_PATH = 'dataset/dogs'
MATCHES_FOLDER = 'static/matches'
RESULTS_FOLDER = 'static/results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(MATCHES_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATASET_PATH'] = DATASET_PATH
app.config['MATCHES_FOLDER'] = MATCHES_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'iniyan1842006',
    'database': 'pupaid'
}

# Load NLP model
try:
    nlp = spacy.load('en_core_web_md')
except Exception:
    print("Spacy model not found. Installing...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load('en_core_web_md')

DISEASE_DB = {
    'Deep Pyoderma folliculitis or Furunculosis': {
        'common_name': 'Deep Skin Infection',
        'description': 'Severe bacterial infection reaching deep skin layers with painful nodules and draining tracts',
        'symptoms': 'Painful nodules, draining tracts, hair loss, redness, swelling',
        'prevention': [
            'Treat superficial pyoderma promptly before it progresses',
            'Use Elizabethan collar at first signs of excessive licking',
            'Properly manage underlying conditions like allergies or hypothyroidism'
        ],
        'home_remedies': [
            'Epsom salt compress: 1/4 cup in 2 quarts warm water, apply for 10 mins 3x/day',
            'Medical-grade honey: Apply under vet-directed bandage changes',
            'Turmeric paste: 1 tsp turmeric + 1 tsp coconut oil + pinch black pepper'
        ],
        'recommended_medicine': [
            'Cefovecin injection: 8 mg/kg SC every 1-2 weeks',
            'Enrofloxacin: 5-10 mg/kg PO SID (not for growing puppies)',
            'Silver sulfadiazine 1% cream: Apply under non-stick bandage changed daily'
        ],
        'severity': 'High',
        'contagious': 'No',
        'keywords': ['deep infection', 'painful nodules', 'draining tracts', 'severe', 'furunculosis']
    },
    'demodectic mange': {
        'common_name': 'Red Mange',
        'description': 'Non-contagious mite infestation (Demodex canis) causing patchy hair loss',
        'symptoms': 'Patchy hair loss, redness, scaling, sometimes itching',
        'prevention': [
            'Avoid breeding animals with chronic generalized demodicosis',
            'Maintain good overall health to support immune system',
            'Regular parasite prevention for all pets in household'
        ],
        'home_remedies': [
            'Omega-3 supplements: 500mg fish oil per 20lbs daily',
            'Probiotics: Give daily to support immune function',
            'Aloe vera: Apply to soothe irritated skin'
        ],
        'recommended_medicine': [
            'Isoxazolines: Fluralaner (Bravecto) or sarolaner (Simparica) per label',
            'Ivermectin: 0.3-0.6 mg/kg PO SID (with MDR1 testing in susceptible breeds)',
            'Amitraz dips: 0.025-0.05% solution applied weekly by vet'
        ],
        'severity': 'Moderate',
        'contagious': 'No',
        'keywords': ['mange', 'demodex', 'hair loss', 'patchy', 'redness']
    },
    'Folliculitis': {
        'common_name': 'Bacterial Follicle Infection',
        'description': 'Superficial bacterial infection of hair follicles causing pustules',
        'symptoms': 'Pustules, redness, hair loss, itching, scaling',
        'prevention': [
            'Weekly antiseptic baths with chlorhexidine 2-4%',
            'Regular grooming to prevent matted hair',
            'Address underlying allergies or endocrine disorders'
        ],
        'home_remedies': [
            'Warm saline compress: 1 tsp salt per cup water, apply 5 mins 2x/day',
            'Aloe vera gel: Apply pure gel to soothe inflammation',
            'Diluted tea tree oil: 3 drops in 1 tbsp coconut oil'
        ],
        'recommended_medicine': [
            'Cephalexin: 22-30 mg/kg PO BID for minimum 21 days',
            'Mupirocin 2% ointment: Apply to lesions BID with glove',
            'Chlorhexidine 2% rinse: Sponge on affected areas daily'
        ],
        'severity': 'Low',
        'contagious': 'No',
        'keywords': ['folliculitis', 'pustules', 'redness', 'itching', 'superficial']
    },
    'Malassezia Dermatitis or yeast dermatitis': {
        'common_name': 'Yeast Infection',
        'description': 'Overgrowth of Malassezia yeast causing greasy, smelly skin',
        'symptoms': 'Greasy skin, strong odor, redness, itching, dark pigmentation',
        'prevention': [
            'Bi-weekly antifungal baths during humid months',
            'Weekly ear cleaning with drying solutions',
            'Feed low-carb diet to reduce yeast food sources'
        ],
        'home_remedies': [
            'Apple cider vinegar rinse: 1/4 cup in 1 quart water after baths',
            'Coconut oil: Apply thin layer as natural antifungal',
            'Probiotic supplements: Give daily with meals'
        ],
        'recommended_medicine': [
            'Ketoconazole: 5-10 mg/kg PO SID with fatty meal',
            'Miconazole/chlorhexidine wipes: Clean affected areas BID',
            'Fluconazole: 2.5-5 mg/kg PO SID for resistant cases'
        ],
        'severity': 'Moderate',
        'contagious': 'No',
        'keywords': ['yeast', 'malassezia', 'greasy', 'odor', 'itchy']
    },
    'mange': {
        'common_name': 'Sarcoptic Mange',
        'description': 'Highly contagious mite infestation (Sarcoptes scabiei) causing intense itching',
        'symptoms': 'Intense itching, hair loss, crusting, redness, secondary infections',
        'prevention': [
            'Monthly isoxazoline preventives (NexGard, Simparica)',
            'Wash all bedding weekly in hot water with borax',
            'Isolate infected pets immediately'
        ],
        'home_remedies': [
            'Neem oil spray: 10 drops in 1 cup water + 1 tsp mild soap',
            'Oatmeal baths: Blend 1 cup oatmeal, mix in bathwater, soak 10 mins',
            'Sulfur lime dips: Veterinary formula applied weekly'
        ],
        'recommended_medicine': [
            'Sarolaner: 2-4 mg/kg PO monthly',
            'Ivermectin: 0.3 mg/kg SC every 2 weeks (off-label)',
            'Selamectin: Topical application every 2 weeks'
        ],
        'severity': 'High',
        'contagious': 'Yes',
        'keywords': ['mange', 'sarcoptic', 'itching', 'contagious', 'crusting']
    },
    'Pyoderma': {
        'common_name': 'Superficial Skin Infection',
        'description': 'Bacterial infection causing pustules and epidermal collarettes',
        'symptoms': 'Pustules, circular crusts (collarettes), redness, itching',
        'prevention': [
            'Daily cleaning of skin folds in prone breeds',
            'Maintain ideal body weight',
            'Keep skin dry - towel thoroughly after swimming'
        ],
        'home_remedies': [
            'Warm saline soaks: 2 tsp sea salt + 1 tsp baking soda per cup water',
            'Manuka honey: Apply thin layer to affected areas',
            'Povidone-iodine: 1:10 dilution as spot treatment'
        ],
        'recommended_medicine': [
            'Cephalexin: 22 mg/kg PO BID for 3 weeks',
            'Chlorhexidine/miconazole shampoo: Lather for 10 mins twice weekly',
            'Chloramphenicol: 40-50 mg/kg PO TID for resistant cases'
        ],
        'severity': 'Moderate',
        'contagious': 'No',
        'keywords': ['pyoderma', 'bacterial', 'pustules', 'collarettes', 'redness']
    },
    'ringworm': {
        'common_name': 'Dermatophytosis',
        'description': 'Fungal infection causing circular hair loss and scaling (not a worm)',
        'symptoms': 'Circular hair loss, scaling, redness, sometimes itching',
        'prevention': [
            'Vacuum floors daily during treatment',
            'Allow sunlight exposure to bedding',
            'Restrict access to carpeted areas during infection'
        ],
        'home_remedies': [
            'Apple cider vinegar: 1:3 dilution applied twice daily',
            'Coconut oil: Massage into affected areas',
            'Sun therapy: 10-15 mins direct sunlight on lesions'
        ],
        'recommended_medicine': [
            'Itraconazole: 5 mg/kg PO SID with fatty food for 4-6 weeks',
            'Lime sulfur 2% dips: Apply weekly until cured',
            'Terbinafine cream: Apply beyond lesion margins BID'
        ],
        'severity': 'Moderate',
        'contagious': 'Yes',
        'keywords': ['ringworm', 'fungal', 'circular', 'hair loss', 'contagious']
    },
    'yeast': {
        'common_name': 'Yeast Overgrowth',
        'description': 'Malassezia pachydermatis overgrowth causing greasy, malodorous skin',
        'symptoms': 'Greasy skin, strong odor, redness, itching, dark pigmentation',
        'prevention': [
            'Weekly antifungal baths for predisposed breeds',
            'Regular ear cleaning with drying solutions',
            'Low-glycemic diet to starve yeast'
        ],
        'home_remedies': [
            'Apple cider vinegar rinse: 1/4 cup in 1 quart water after bathing',
            'Coconut oil: Contains caprylic acid - apply thin layer',
            'Probiotics: Give daily to support healthy microbiome'
        ],
        'recommended_medicine': [
            'Ketoconazole: 5-10 mg/kg PO SID with acidic food',
            'Miconazole 2% + chlorhexidine 2% shampoo: Lather for 10 mins twice weekly',
            'Fluconazole: 2.5-5 mg/kg PO SID for systemic treatment'
        ],
        'severity': 'Moderate',
        'contagious': 'No',
        'keywords': ['yeast', 'malassezia', 'greasy', 'odor', 'itchy']
    }
}


class SkinInfectionDetector:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.feature_extractor = None
        self.train_features = None
        self.y_encoded = None
        self.le = None
        self.train_image_paths = None
        self.X = None
        self.y = None
        self.model_trained = False
        self.initialize_gpu()
        self.train_model()

    def initialize_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Using GPU: {gpus[0].name}")
            except RuntimeError as e:
                print(f"Error configuring GPU: {e}")
        else:
            print("No GPU available, using CPU")

    def load_and_preprocess_image(self, image_path, target_size=(224, 224)):
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            img = cv2.resize(img, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_input(img)
            return img
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def load_dataset_with_paths(self):
        images = []
        labels = []
        image_paths = []

        if not os.path.exists(self.dataset_path):
            print(f"Dataset path not found: {self.dataset_path}")
            return np.array(images), np.array(labels), image_paths

        print("Loading dataset...")
        for label in os.listdir(self.dataset_path):
            label_path = os.path.join(self.dataset_path, label)
            if os.path.isdir(label_path):
                print(f"Processing category: {label}")
                for image_file in os.listdir(label_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(label_path, image_file)
                        img = self.load_and_preprocess_image(image_path)
                        if img is not None:
                            images.append(img)
                            labels.append(label)
                            image_paths.append(image_path)
        print(f"Total images loaded: {len(images)}")
        return np.array(images), np.array(labels), image_paths

    def create_feature_extractor(self):
        print("Creating feature extractor...")
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=(224, 224, 3), pooling='avg')
        model = Model(inputs=base_model.input, outputs=base_model.output)
        return model

    def train_model(self):
        print("Loading dataset...")
        self.X, self.y, self.train_image_paths = self.load_dataset_with_paths()

        if len(self.X) == 0:
            print("No valid images found in dataset. Training aborted.")
            return

        print("Creating feature extractor...")
        self.feature_extractor = self.create_feature_extractor()

        print("Extracting features...")
        self.train_features = self.feature_extractor.predict(self.X, verbose=1)

        print("Encoding labels...")
        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(self.y)

        self.model_trained = True
        print("Model training complete!")

    def predict_infection(self, test_image_path):
        if not self.model_trained:
            return {'error': 'Model not trained'}

        print(f"Processing test image: {test_image_path}")
        processed_img = self.load_and_preprocess_image(test_image_path)
        if processed_img is None:
            return {'error': 'Could not process test image'}

        test_feature = self.feature_extractor.predict(
            np.expand_dims(processed_img, axis=0), verbose=0)

        similarities = cosine_similarity(test_feature, self.train_features)
        max_idx = np.argmax(similarities)
        confidence = float(similarities[0, max_idx])
        predicted_label = self.le.inverse_transform([self.y_encoded[max_idx]])[0]

        # Get disease information
        disease_info = DISEASE_DB.get(predicted_label, {})
        if not disease_info:
            disease_info = {
                'name': predicted_label,
                'description': 'No description available',
                'symptoms': 'No symptom information',
                'prevention': ['Consult veterinarian for prevention strategies'],
                'home_remedies': ['Consult veterinarian for home care'],
                'recommended_medicine': ['Consult veterinarian for treatment options'],
                'severity': 'Unknown',
                'contagious': 'Unknown'
            }
        else:
            disease_info = disease_info.copy()
            disease_info['name'] = predicted_label

        # Create comprehensive report
        report = self.create_disease_report(disease_info, confidence)

        # Create result visualization
        result_img_name = f"result_{uuid.uuid4()}.png"
        result_img_path = os.path.join(app.config['RESULTS_FOLDER'], result_img_name)
        self.create_result_visualization(test_image_path, predicted_label, confidence, result_img_path)

        return {
            'prediction': predicted_label,
            'confidence': confidence,
            'report': report,
            'result_image': f"/static/results/{result_img_name}",
            'disease_info': disease_info,
            'uploaded_image': f"/static/uploads/{os.path.basename(test_image_path)}"
        }

    def create_disease_report(self, disease_info, confidence):
        report = f"\n{'=' * 80}\n"
        report += "CANINE SKIN INFECTION DIAGNOSTIC REPORT\n"
        report += f"{'=' * 80}\n\n"
        report += f"CONDITION: {disease_info['name'].upper()} (Confidence: {confidence:.1%})\n\n"

        report += f"DESCRIPTION:\n{disease_info['description']}\n\n"

        report += "SYMPTOMS:\n"
        report += disease_info['symptoms'] + "\n\n"

        report += "PREVENTION STRATEGIES:\n"
        for i, item in enumerate(disease_info['prevention'], 1):
            report += f"{i}. {item}\n"
        report += "\n"

        report += "HOME REMEDIES:\n"
        for i, item in enumerate(disease_info['home_remedies'], 1):
            report += f"{i}. {item}\n"
        report += "\n"

        report += "RECOMMENDED MEDICINE:\n"
        for i, item in enumerate(disease_info['recommended_medicine'], 1):
            report += f"{i}. {item}\n"
        report += "\n"

        report += f"SEVERITY: {disease_info['severity']}\n"
        report += f"CONTAGIOUS: {disease_info['contagious']}\n"
        report += f"{'=' * 80}"

        return report

    def create_result_visualization(self, test_img_path, label, confidence, output_path):
        test_img = cv2.cvtColor(cv2.imread(test_img_path), cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(15, 8))

        # Test image
        plt.subplot(1, 2, 1)
        plt.imshow(test_img)
        plt.title("Uploaded Image")
        plt.axis('off')

        # Confidence and disease info
        plt.subplot(1, 2, 2)
        plt.text(0.05, 0.8, f"Prediction: {label}\nConfidence: {confidence:.2%}",
                 fontsize=14, fontweight='bold')
        plt.text(0.05, 0.5, "Key Information:", fontsize=12, fontweight='bold')
        plt.text(0.05, 0.4, f"Description: {DISEASE_DB[label]['description'][:100]}...", fontsize=10)
        plt.text(0.05, 0.3, f"Severity: {DISEASE_DB[label]['severity']}", fontsize=10)
        plt.text(0.05, 0.2, f"Contagious: {DISEASE_DB[label]['contagious']}", fontsize=10)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()


def process_text_query(query):
    # Preprocess the query
    doc = nlp(query.lower())
    query_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    # Prepare disease data for comparison
    disease_texts = []
    disease_names = []
    for name, info in DISEASE_DB.items():
        text = ' '.join([
            name.lower(),
            info['description'].lower(),
            info['symptoms'].lower(),
            ' '.join(info['keywords'])
        ])
        disease_texts.append(text)
        disease_names.append(name)

    # Vectorize and compare
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(disease_texts)
        query_vec = vectorizer.transform([' '.join(query_tokens)])

        # Calculate similarities
        similarities = text_cosine_similarity(query_vec, tfidf_matrix)
        sorted_indices = np.argsort(similarities[0])[::-1]

        # Get top matches
        matches = []
        for idx in sorted_indices[:3]:  # Top 3 matches
            if similarities[0][idx] > 0.1:  # Minimum similarity threshold
                matches.append({
                    'condition': disease_names[idx],
                    'confidence': float(similarities[0][idx]),
                    'description': DISEASE_DB[disease_names[idx]]['description']
                })

        # Generate recommendations based on query
        recommendations = [
            "Monitor your dog's symptoms closely",
            "Consider consulting a veterinarian for proper diagnosis",
            "Keep the affected area clean and dry"
        ]

        # Add specific recommendations if certain keywords are found
        if any(word in query_tokens for word in ['itch', 'itching', 'scratch']):
            recommendations.append("Consider using an anti-itch shampoo or spray")

        if any(word in query_tokens for word in ['red', 'redness', 'inflam']):
            recommendations.append("Apply a cool compress to reduce inflammation")

        return {
            'matches': matches,
            'recommendations': recommendations
        }
    except Exception as e:
        print(f"Error processing text query: {e}")
        return {'error': 'Could not process your query'}


# Database connection helper
def get_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None


# Initialize detector
print("Initializing skin infection detector...")
detector = SkinInfectionDetector(DATASET_PATH)


@app.route('/')
@app.route('/index')
@app.route('/index.html')
def index():
    # Check if user is logged in
    if 'user_id' not in session:
        return render_template('signin.html')

    # Check if user has dog info
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            cursor.execute("SELECT * FROM pets WHERE user_id = %s", (session['user_id'],))
            pet = cursor.fetchone()
            cursor.close()
            connection.close()

            if pet:
                return render_template('index.html')
            else:
                return render_template('dog-info.html')
        except Error as e:
            print(f"Error checking pet info: {e}")
            return render_template('signin.html')
    return render_template('signin.html')


@app.route('/login')
@app.route('/signin.html')
def login_page():
    return render_template('signin.html')


@app.route('/register')
@app.route('/register.html')
def register_page():
    return render_template('register.html')


@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        # Check if user exists in database
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
                user = cursor.fetchone()

                if user and check_password_hash(user['password'], password):
                    # Set session variables
                    session['user_id'] = user['id']
                    session['email'] = user['email']

                    # Check if user has pet information
                    cursor.execute("SELECT * FROM pets WHERE user_id = %s", (user['id'],))
                    pet = cursor.fetchone()

                    cursor.close()
                    connection.close()

                    return jsonify({
                        'success': True,
                        'hasDogInfo': pet is not None
                    })
                else:
                    cursor.close()
                    connection.close()
                    return jsonify({'error': 'Invalid email or password'}), 401

            except Error as e:
                print(f"Database error: {e}")
                if connection:
                    connection.close()
                return jsonify({'error': 'Database error. Please try again.'}), 500
        else:
            return jsonify({'error': 'Database connection failed. Please try again.'}), 500

    except Exception as e:
        print(f"Login error: {str(e)}")
        return jsonify({'error': 'Login failed. Please try again.'}), 500


@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email', '').strip()
        password = data.get('password', '').strip()
        confirm_password = data.get('confirm_password', '').strip()

        # Validation
        if not email or not password or not confirm_password:
            return jsonify({'error': 'All fields are required'}), 400

        if password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400

        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify({'error': 'Invalid email format'}), 400

        # Check if user already exists
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
                existing_user = cursor.fetchone()

                if existing_user:
                    cursor.close()
                    connection.close()
                    return jsonify({'error': 'Email already registered'}), 400

                # Create new user
                hashed_password = generate_password_hash(password)
                cursor.execute(
                    "INSERT INTO users (email, password) VALUES (%s, %s)",
                    (email, hashed_password)
                )
                connection.commit()

                # Get the new user ID
                user_id = cursor.lastrowid
                cursor.close()
                connection.close()

                # Set session variables
                session['user_id'] = user_id
                session['email'] = email

                return jsonify({
                    'success': True,
                    'message': 'Registration successful',
                    'hasDogInfo': False
                })

            except Error as e:
                print(f"Database error: {e}")
                if connection:
                    connection.close()
                return jsonify({'error': 'Database error. Please try again.'}), 500
        else:
            return jsonify({'error': 'Database connection failed. Please try again.'}), 500

    except Exception as e:
        print(f"Registration error: {str(e)}")
        return jsonify({'error': 'Registration failed. Please try again.'}), 500


@app.route('/dog-info')
@app.route('/dog-info.html')
def dog_info_page():
    if 'user_id' not in session:
        return render_template('signin.html')
    return render_template('dog-info.html')


@app.route('/save-dog-info', methods=['POST'])
def save_dog_info():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Please login first'}), 401

        data = request.get_json()
        dog_name = data.get('dog_name', '').strip()
        dog_age = data.get('dog_age')
        dog_weight = data.get('dog_weight')
        dog_breed = data.get('dog_breed', '').strip()

        if not dog_name or dog_age is None or dog_weight is None or not dog_breed:
            return jsonify({'error': 'All fields are required'}), 400

        # Validate age and weight
        if dog_age < 0 or dog_age > 25:
            return jsonify({'error': 'Age must be between 0 and 25 years'}), 400

        if dog_weight < 1 or dog_weight > 200:
            return jsonify({'error': 'Weight must be between 1 and 200 lbs'}), 400

        # Save dog information to database
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()

                # Check if pet already exists for this user
                cursor.execute(
                    "SELECT id FROM pets WHERE user_id = %s",
                    (session['user_id'],)
                )
                existing_pet = cursor.fetchone()

                if existing_pet:
                    # Update existing pet
                    cursor.execute(
                        "UPDATE pets SET name = %s, breed = %s, age = %s, weight = %s WHERE user_id = %s",
                        (dog_name, dog_breed, dog_age, dog_weight, session['user_id'])
                    )
                else:
                    # Insert new pet
                    cursor.execute(
                        "INSERT INTO pets (user_id, name, breed, age, weight) VALUES (%s, %s, %s, %s, %s)",
                        (session['user_id'], dog_name, dog_breed, dog_age, dog_weight)
                    )

                connection.commit()
                cursor.close()
                connection.close()

                return jsonify({
                    'success': True,
                    'message': 'Dog information saved successfully'
                })

            except Error as e:
                print(f"Database error: {e}")
                if connection:
                    connection.close()
                return jsonify({'error': 'Failed to save dog information. Please try again.'}), 500
        else:
            return jsonify({'error': 'Database connection failed. Please try again.'}), 500

    except Exception as e:
        print(f"Save dog info error: {str(e)}")
        return jsonify({'error': 'Failed to save dog information. Please try again.'}), 500


@app.route('/get-dog-info')
def get_dog_info():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Please login first'}), 401

        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor(dictionary=True)
                cursor.execute(
                    "SELECT * FROM pets WHERE user_id = %s",
                    (session['user_id'],)
                )
                pet = cursor.fetchone()
                cursor.close()
                connection.close()

                if pet:
                    return jsonify({
                        'success': True,
                        'dog_info': pet
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No dog information found'
                    }), 404

            except Error as e:
                print(f"Database error: {e}")
                if connection:
                    connection.close()
                return jsonify({'error': 'Failed to retrieve dog information.'}), 500
        else:
            return jsonify({'error': 'Database connection failed. Please try again.'}), 500

    except Exception as e:
        print(f"Get dog info error: {str(e)}")
        return jsonify({'error': 'Failed to retrieve dog information.'}), 500


@app.route('/diagnosis')
@app.route('/diagnosis.html')
def diagnosis():
    if 'user_id' not in session:
        return render_template('signin.html')
    return render_template('diagnosis.html')


@app.route('/text-diagnosis')
@app.route('/text-diagnosis.html')
def text_diagnosis():
    if 'user_id' not in session:
        return render_template('signin.html')
    return render_template('text-diagnosis.html')


@app.route('/schedule')
@app.route('/schedule.html')
def schedule():
    if 'user_id' not in session:
        return render_template('signin.html')
    return render_template('schedule.html')


@app.route('/logout')
def logout():
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Please login first'}), 401

        if 'file' not in request.files:
            # Check if the file is coming as raw data (from camera)
            if not request.data:
                return jsonify({'error': 'No file uploaded'}), 400

            # Handle raw image data from camera
            file_data = request.data
            filename = f"{uuid.uuid4()}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            with open(filepath, 'wb') as f:
                f.write(file_data)
        else:
            # Handle regular file upload
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400

            # Save uploaded file
            filename = f"{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

        # Process image
        result = detector.predict_infection(filepath)
        if 'error' in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return jsonify({'error': 'Processing failed. Please try another image.'}), 500


@app.route('/search', methods=['POST'])
def search():
    try:
        if 'user_id' not in session:
            return jsonify({'error': 'Please login first'}), 401

        data = request.get_json()
        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Empty query'}), 400

        # Save search query to database
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute(
                    "INSERT INTO searches (user_id, query) VALUES (%s, %s)",
                    (session['user_id'], query)
                )
                connection.commit()
                cursor.close()
                connection.close()
            except Error as e:
                print(f"Error saving search query: {e}")
                # Continue processing even if saving fails

        results = process_text_query(query)
        return jsonify(results)

    except Exception as e:
        print(f"Error during search: {str(e)}")
        return jsonify({'error': 'Search failed. Please try again.'}), 500


# Serve static files
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads', filename)


@app.route('/static/results/<filename>')
def result_file(filename):
    return send_from_directory('static/results', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)