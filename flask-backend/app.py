from flask import Flask, request, jsonify
from pymongo import MongoClient
import os
from transformers import pipeline
import PyPDF2
from docx import Document
from flask_cors import CORS
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['resume_analyzer']
collection = db['resumes']

# Set up file upload directory
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Helper function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise e

# Helper function to extract text from DOCX
def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text
        return text
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {e}")
        raise e

@app.route('/analyze', methods=['POST'])
def analyze_resume():
    try:
        # Check if the necessary data is provided
        if 'resume' not in request.files or 'job_description' not in request.form:
            logging.error("Invalid request, missing resume or job description")
            return jsonify({'message': 'Invalid request, missing resume or job description'}), 400

        # Save the uploaded resume file
        resume_file = request.files['resume']
        if not resume_file.filename.endswith(('.txt', '.pdf', '.docx')):
            logging.error("Invalid file type. Only TXT, PDF, or DOCX files are allowed.")
            return jsonify({'message': 'Invalid file type. Only TXT, PDF, or DOCX files are allowed.'}), 400

        resume_path = os.path.join(UPLOAD_FOLDER, resume_file.filename)
        resume_file.save(resume_path)

        # Read and process the resume text
        if resume_file.filename.endswith('.txt'):
            with open(resume_path, 'r', encoding='utf-8') as file:
                resume_text = file.read()
        elif resume_file.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume_path)
        elif resume_file.filename.endswith('.docx'):
            resume_text = extract_text_from_docx(resume_path)

        # Check if resume text is empty
        if not resume_text.strip():
            logging.error("No text extracted from resume")
            return jsonify({'message': 'Failed to extract text from resume'}), 400

        # Get the job description from the form
        job_description = request.form['job_description']
        if not job_description.strip():
            logging.error("Job description is empty")
            return jsonify({'message': 'Job description is empty'}), 400

        # Log the extracted texts (for debugging purposes)
        logging.debug(f"Resume Text (Preview): {resume_text[:200]}")
        logging.debug(f"Job Description (Preview): {job_description[:200]}")

        # Analyze the match between the resume and job description
        result = analyze_match(resume_text, job_description)
        return jsonify({'message': result})  # Ensure this is the right response format

    except Exception as e:
        logging.error(f"Error in /analyze endpoint: {e}")
        return jsonify({'message': f"Internal Server Error: {str(e)}"}), 500

@app.route('/save', methods=['POST'])
def save_to_db():
    try:
        data = request.json
        collection.insert_one(data)  # Insert the data into the MongoDB collection
        return jsonify({'message': 'Data saved successfully!'})
    except Exception as e:
        logging.error(f"Error saving data to MongoDB: {e}")
        return jsonify({'message': f"Error saving data: {str(e)}"}), 500

def analyze_match(resume, job_description):
    try:
        # Create a TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()

        # Combine resume and job description texts into a list
        texts = [resume, job_description]

        # Fit and transform the texts to calculate TF-IDF
        tfidf_matrix = vectorizer.fit_transform(texts)

        # Calculate cosine similarity between the two texts
        similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Log the cosine similarity score (for debugging purposes)
        logging.debug(f"Cosine Similarity Score: {similarity_score}")

        # Convert the similarity score to a percentage
        match_percentage = round(similarity_score * 100, 2)

        # Return the match percentage along with the message
        result = f"Resume relevance to job description: {match_percentage}%"
        logging.debug(f"Final result: {result}")  # Add logging for debugging
        return result
    except Exception as e:
        logging.error(f"Error analyzing match: {e}")
        raise e

if __name__ == '__main__':
    app.run(debug=True)
