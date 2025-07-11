# DiaFriend - Diabetes Management Chatbot

This Flask app provides a simple, friendly chatbot to help diabetic patients:

- Log daily blood sugar readings
- Ask diabetes-related questions
- Set reminders

It uses OpenAI GPT-4 to generate clear, supportive responses while **avoiding medical advice beyond general guidance**.

---

## 🚀 How to Run

1. **Install dependencies:**

   ```bash
   pip install flask python-dotenv openai

## Set your OpenAI API key:

Create a .env file in the project folder:

OPENAI_API_KEY=your_api_key_here

## Access in your browser:

http://localhost:5000

## API Endpoints
POST /chat

{
  "message": "your message",
  "thread_id": "optional_thread_id"
}
Returns the chatbot reply and conversation state.

POST /reset_conversation

Resets the conversation memory.