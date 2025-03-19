from flask import Flask, render_template, request, jsonify
from gaming_chatbot import GamingChatbot
import uuid

app = Flask(__name__)
chatbot = GamingChatbot()

# Store user sessions
user_sessions = {}

@app.route('/')
def index():
    """Render the chat interface"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Process chat messages and return response"""
    data = request.json
    message = data.get('message', '')
    
    # Get or create user session
    session_id = request.cookies.get('user_session')
    if not session_id or session_id not in user_sessions:
        session_id = str(uuid.uuid4())
        user_sessions[session_id] = True
    
    # Get response from chatbot
    response = chatbot.respond(session_id, message)
    
    return jsonify({
        'response': response,
        'session_id': session_id
    })

if __name__ == '__main__':
    app.run(debug=True) 