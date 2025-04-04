from flask import Flask, request, jsonify
from flask_cors import CORS
from gaming_chatbot import GamingChatbot
import uuid

app = Flask(__name__)

# Allow requests from all origins
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize the chatbot
chatbot = GamingChatbot()

# Store active sessions
sessions = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message')
        session_id = data.get('session_id')
        
        # Validate input
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Create new session if none exists
        if not session_id:
            session_id = str(uuid.uuid4())
            sessions[session_id] = {'created_at': uuid.uuid1().hex}
        
        # Get response from chatbot
        response = chatbot.respond(session_id, message)
        
        return jsonify({
            'response': response,
            'session_id': session_id
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/new-session', methods=['POST'])
def new_session():
    try:
        session_id = str(uuid.uuid4())
        sessions[session_id] = {'created_at': uuid.uuid1().hex}
        return jsonify({'session_id': session_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
