from flask import Flask, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
import re
import json
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter

# ✅ Set OpenAI API key here
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client (new style)
client = OpenAI()

# Function to extract YouTube video ID
def get_video_id(url):
    patterns = [
        r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:embed/)([a-zA-Z0-9_-]{11})",
        r"(?:watch\?v=)([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Split long transcript into chunks
def chunk_text(text, max_chunk_size=3000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0

    for word in words:
        if current_size + len(word) > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
            current_size += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

@app.route('/generate_qa', methods=['POST'])
def generate_qa():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        url = data.get('url', '').strip()
        count = int(data.get('count', 10))

        if not url:
            return jsonify({'error': 'YouTube URL is required'}), 400

        if count < 1 or count > 50:
            return jsonify({'error': 'Question count must be between 1 and 50'}), 400

        video_id = get_video_id(url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL format'}), 400

        # Get transcript
        try:
            ytt = YouTubeTranscriptApi()
            transcript = ytt.fetch(video_id, languages=['en', 'bn', 'hi'])

        except Exception as e:
            return jsonify({'error': f"Could not fetch transcript: {str(e)}"}), 500

        full_text = " ".join([entry.text for entry in transcript])

        if len(full_text.strip()) < 100:
            return jsonify({'error': 'Transcript too short to generate meaningful questions'}), 400

        chunks = chunk_text(full_text)
        text_to_process = chunks[0]

        # Prompt to OpenAI
        prompt = f"""Based on the following transcript, generate exactly {count} educational question-answer pairs in JSON format.

Requirements:
- Return ONLY a valid JSON array
- Each question should be clear and educational
- Each answer should be concise but complete
- Focus on key concepts and important information

Format:
[
  {{"question": "What is...?", "answer": "The answer is..."}},
  {{"question": "How does...?", "answer": "It works by..."}}
]

Transcript:
{text_to_process}"""

        # New OpenAI API call style
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an educational content generator. Always return valid JSON arrays only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.7
        )

        result = response.choices[0].message.content.strip()

        try:
            json_result = json.loads(result)
            if isinstance(json_result, list):
                return jsonify({'result': result, 'count': len(json_result)})
            else:
                return jsonify({'error': 'Invalid response format from AI'}), 500
        except json.JSONDecodeError:
            return jsonify({'error': 'AI returned invalid JSON format'}), 500

    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Server is running', 'openai_configured': bool(openai_api_key)})

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Flask app is running on Render'})

if __name__ == '__main__':
    # Use PORT environment variable provided by Render
    port = int(os.environ.get('PORT', 5000))
    if not openai_api_key:
        print("⚠️ Warning: OPENAI_API_KEY not set!")
    app.run(host='0.0.0.0', port=port, debug=False)