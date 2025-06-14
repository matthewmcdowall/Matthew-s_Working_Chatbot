# AI Chatbot with Memory - Production Ready

A full-featured AI chatbot built with Flask and LangChain that maintains conversation memory and is ready for deployment on Render.

## Features

- 🤖 **Smart AI Conversations** - Powered by OpenAI's GPT models
- 🧠 **Conversation Memory** - Remembers context within each session
- 🎨 **Modern Web Interface** - Responsive, mobile-friendly design
- 🚀 **Production Ready** - Configured for easy deployment
- 🔒 **Secure** - Proper environment variable handling
- 📱 **Mobile Responsive** - Works great on all devices

## Quick Deploy to Render

### Option 1: Deploy with GitHub (Recommended)

1. **Fork this repository** to your GitHub account

2. **Get OpenAI API Key**
   - Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
   - Create a new API key
   - Copy the key (starts with `sk-`)

3. **Deploy on Render**
   - Go to [Render](https://render.com) and sign up/login
   - Click "New +" → "Web Service"
   - Connect your GitHub account and select this repository
   - Use these settings:
     ```
     Name: ai-chatbot (or your preferred name)
     Environment: Python 3
     Build Command: pip install -r requirements.txt
     Start Command: gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
     ```

4. **Set Environment Variables**
   - In your Render service dashboard, go to "Environment"
   - Add these variables:
     ```
     OPENAI_API_KEY = your_openai_api_key_here
     SECRET_KEY = any_random_string_here
     FLASK_ENV = production
     ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment (usually 2-3 minutes)
   - Your chatbot will be live at `https://your-app-name.onrender.com`

### Option 2: Deploy with Docker

If you prefer using the Dockerfile:

1. Make sure `Dockerfile` is in your repository root
2. In Render, select "Docker" as the environment
3. Set the same environment variables as above

## Local Development

### Prerequisites

- Python 3.11+
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ai-chatbot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   # Copy the example file
   cp .env.example .env
   
   # Edit .env with your API key
   OPENAI_API_KEY=your_actual_api_key_here
   SECRET_KEY=your_secret_key_here
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   - Go to `http://localhost:5000`
   - Start chatting!

## Project Structure

```
ai-chatbot/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── render.yaml           # Render deployment config
├── .env.example          # Environment variables template
├── templates/
│   └── index.html        # Chat web interface
└── README.md             # This file
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Your OpenAI API key | Yes |
| `SECRET_KEY` | Flask session secret | Yes |
| `FLASK_ENV` | Environment (production/development) | No |
| `PORT` | Port to run on (set by Render) | No |

### Memory Configuration

The chatbot uses `ConversationBufferWindowMemory` with a window of 10 exchanges. You can modify this in `app.py`:

```python
memory = ConversationBufferWindowMemory(
    k=10,  # Number of exchanges to remember
    return_messages=True
)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Chat interface |
| `/api/chat` | POST | Send message |
| `/api/clear` | POST | Clear conversation |
| `/api/health` | GET | Health check |

## Customization

### Changing the AI Model

Edit the model in `app.py`:

```python
self.llm = ChatOpenAI(
    model="gpt-4",  # Change to gpt-4, gpt-3.5-turbo, etc.
    temperature=0.7,
    openai_api_key=self.openai_api_key
)
```

### Styling the Interface

Modify the CSS in `templates/index.html` to change the appearance.

### Adding Features

The codebase is modular and easy to extend:
- Add new API endpoints in `app.py`
- Enhance the memory system with databases
- Add user authentication
- Implement conversation persistence

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY not found"**
   - Make sure you've set the environment variable in Render
   - Check that your API key is valid

2. **"Module not found" errors**
   - Ensure all dependencies are in `requirements.txt`
   - Try rebuilding on Render

3. **Chat not responding**
   - Check the browser console for errors
   - Verify the API key has credits
   - Check Render logs for server errors

4. **Deployment fails**
   - Check that all files are committed to GitHub
   - Verify the start command is correct
   - Check Render build logs

### Getting Help

- Check Render logs: In your service dashboard → "Logs"
- Check browser console: Press F12 → "Console" tab
- Verify API key: Test at [OpenAI Playground](https://platform.openai.com/playground)

## Scaling and Production

### For High Traffic

1. **Upgrade Render plan** for better performance
2. **Add Redis** for session storage:
   ```python
   # Replace in-memory conversations dict
   import redis
   r = redis.Redis(host='localhost', port=6379, db=0)
   ```

3. **Add rate limiting** to prevent abuse
4. **Implement user authentication** for personalized experiences
5. **Add logging and monitoring** for production insights

### Security Considerations

- API keys are handled securely via environment variables
- Session management prevents conversation mixing
- CORS is configured for web access
- Input validation prevents malicious requests

## License

This project is open source and available under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Render documentation
3. Check OpenAI API status
4. Create an issue in this repository

---

**Happy chatting! 🤖💬**