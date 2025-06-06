import os
import uuid
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.schema import BaseMessage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-change-this')
CORS(app)

# Store conversations in memory (in production, use Redis or database)
conversations = {}

class ChatbotManager:
    def __init__(self):
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm = ChatOpenAI(
            model="gpt-4.0-mini",
            temperature=1,
            openai_api_key=self.openai_api_key
        )
    
    def get_conversation(self, session_id):
        """Get or create a conversation for a session"""
        if session_id not in conversations:
            memory = ConversationBufferWindowMemory(
                k=10,  # Remember last 10 exchanges
                return_messages=True
            )
            conversations[session_id] = ConversationChain(
                llm=self.llm,
                memory=memory,
                verbose=False
            )
        return conversations[session_id]
    
    def chat(self, session_id, message):
        """Send message and get response"""
        try:
            conversation = self.get_conversation(session_id)
            response = conversation.predict(input=message)
            return {"success": True, "response": response}
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def clear_conversation(self, session_id):
        """Clear conversation history"""
        if session_id in conversations:
            del conversations[session_id]
        return {"success": True, "message": "Conversation cleared"}

# Initialize chatbot manager
try:
    chatbot_manager = ChatbotManager()
except ValueError as e:
    logger.error(f"Failed to initialize chatbot: {e}")
    chatbot_manager = None

@app.route('/')
def home():
    """Serve the main chat interface"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    if not chatbot_manager:
        return jsonify({
            "success": False, 
            "error": "Chatbot not initialized. Check OpenAI API key."
        }), 500
    
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({"success": False, "error": "Message is required"}), 400
        
        session_id = session.get('session_id')
        if not session_id:
            session['session_id'] = str(uuid.uuid4())
            session_id = session['session_id']
        
        result = chatbot_manager.chat(session_id, message)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@app.route('/api/clear', methods=['POST'])
def clear_chat():
    """Clear conversation history"""
    if not chatbot_manager:
        return jsonify({"success": False, "error": "Chatbot not initialized"}), 500
    
    session_id = session.get('session_id')
    if session_id:
        result = chatbot_manager.clear_conversation(session_id)
        return jsonify(result)
    
    return jsonify({"success": True, "message": "No conversation to clear"})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "chatbot_ready": chatbot_manager is not None
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)