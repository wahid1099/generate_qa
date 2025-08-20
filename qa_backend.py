from flask import Flask, request, jsonify
import re
import json
import requests
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for Flutter

# API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
youtube_api_key = os.getenv("YOUTUBE_API_KEY")

# Initialize OpenAI client
client = OpenAI()

# ✅ Function to extract YouTube video ID
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

# ✅ Fetch transcript using YouTube Data API
def fetch_transcript(video_id):
    # Step 1: Get captions list
    captions_url = (
        f"https://www.googleapis.com/youtube/v3/captions"
        f"?part=snippet&videoId={video_id}&key={youtube_api_key}"
    )
    captions_response = requests.get(captions_url).json()

    if "items" not in captions_response or not captions_response["items"]:
        return None

    # Pick first available caption track (usually English if available)
    caption_id = captions_response["items"][0]["id"]

    # Step 2: Download transcript (srv3 = XML/JSON structured captions)
    transcript_url = f"https://www.googleapis.com/youtube/v3/captions/{caption_id}?tfmt=srv3&key={youtube_api_key}"
    transcript_response = requests.get(transcript_url)

    if transcript_response.status_code != 200:
        return None

    return transcript_response.text

# ✅ Split transcript into chunks
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

        # ✅ Fetch transcript
        transcript_text = fetch_transcript(video_id)
        if not transcript_text:
            return jsonify({'error': 'Could not fetch transcript'}), 500

        if len(transcript_text.strip()) < 100:
            return jsonify({'error': 'Transcript too short to generate meaningful questions'}), 400

        chunks = chunk_text(transcript_text)
        text_to_process = chunks[0]

        # ✅ Prompt OpenAI
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
    return jsonify({
        'status': 'Server is running',
        'openai_configured': bool(openai_api_key),
        'youtube_configured': bool(youtube_api_key)
    })

if __name__ == '__main__':
    if not openai_api_key:
        print("⚠️ Warning: OPENAI_API_KEY not set!")
    if not youtube_api_key:
        print("⚠️ Warning: YOUTUBE_API_KEY not set!")
    app.run(host='0.0.0.0', port=5000, debug=True)
