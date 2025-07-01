from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import logging
from openai import OpenAI
from uuid import uuid4

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# In-memory thread store with conversation state
threads = {}
conversation_states = {}

SYSTEM_PROMPT = r"""
You are DiaFriend, a professional, warm, and supportive diabetes management assistant.

**Your Role:**
- Help patients manage diabetes daily.
- Answer questions about diet, medication, glucose monitoring, and exercise.
- Log daily blood sugar readings.
- Provide reminders and encouragement.

**Core Principles:**
- Use clear, compassionate language.
- Never provide medical advice beyond general guidance.
- Recommend contacting a healthcare provider for emergencies.
- Keep responses short and supportive.
- Ask only one question at a time.
- Acknowledge the patient's feelings.

**Response Style:**
- Friendly and reassuring.
- Clear and non-judgmental.
- Encouraging.

Always maintain a caring tone and build trust.
"""

def get_conversation_state(thread_id):
    return conversation_states.get(thread_id, {'step': 'initial', 'content_type': None, 'goal': None, 'data': {}})

def update_conversation_state(thread_id, updates):
    if thread_id not in conversation_states:
        conversation_states[thread_id] = {'step': 'initial', 'content_type': None, 'goal': None, 'data': {}}
    conversation_states[thread_id].update(updates)

def get_next_question(state, user_message):
    step = state['step']
    data = state['data']

    if step == 'initial':
        return {
            'message': "👋 Hello! I'm DiaFriend, here to help you manage diabetes. What would you like to do today?",
            'options': ['Log blood sugar reading', 'Ask a diabetes question', 'Set a reminder'],
            'next_step': 'selection_made'
        }

    elif step == 'selection_made':
        selection = user_message.lower()
        if 'log' in selection:
            return {
                'message': "Sure, please enter your blood sugar reading in mg/dL.",
                'next_step': 'log_reading'
            }
        elif 'question' in selection:
            return {
                'message': "Of course! What's your question about diabetes?",
                'next_step': 'ask_question'
            }
        elif 'reminder' in selection:
            return {
                'message': "Okay! What time would you like me to remind you? (e.g., 8:00 AM)",
                'next_step': 'set_reminder'
            }

    elif step == 'log_reading':
        return {
            'message': f"Got it! I've logged your reading: {user_message} mg/dL. Would you like to do anything else?",
            'options': ['Log another reading', 'Ask a question', 'Set a reminder', 'Nothing else'],
            'next_step': 'selection_made'
        }

    elif step == 'set_reminder':
        return {
            'message': f"Reminder set for {user_message}. Is there anything else you’d like help with?",
            'options': ['Log a reading', 'Ask a question', 'Nothing else'],
            'next_step': 'selection_made'
        }

    elif step == 'ask_question':
        return {
            'message': "Thanks for your question. Let me get you some helpful information.",
            'next_step': 'generate_answer'
        }

    return {
        'message': "I'm ready for your next request!",
        'next_step': 'selection_made'
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data.get('message')
    thread_id = data.get('thread_id')

    if not message:
        return jsonify({'error': 'Message is required'}), 400

    if not thread_id or thread_id not in threads:
        thread_id = str(uuid4())
        threads[thread_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    state = get_conversation_state(thread_id)
    next_response = get_next_question(state, message)

    new_state_updates = {
        'step': next_response['next_step'],
        'data': {**state['data'], 'last_response': message}
    }

    update_conversation_state(thread_id, new_state_updates)
    threads[thread_id].append({"role": "user", "content": message})

    # If we're at the final content generation step
    if next_response['next_step'] == 'generate_answer':
        try:
            full_context = get_conversation_state(thread_id)
            question_text = state['data'].get('last_response')
            content_prompt = f"""
You are DiaFriend. Please provide a friendly, clear, and non-judgmental answer to this diabetes-related question:

"{question_text}"

Remember:
- Do not give medical advice beyond general information.
- Encourage contacting a healthcare provider for urgent concerns.
- Keep the answer short and supportive.
"""

            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )

            final_content = response.choices[0].message.content
            threads[thread_id].append({"role": "assistant", "content": final_content})

            return jsonify({
                'response': final_content,
                'thread_id': thread_id,
                'conversation_state': get_conversation_state(thread_id)
            })

        except Exception as e:
            logger.error(f"Error during content generation: {e}")
            return jsonify({'error': str(e)}), 500

    # Otherwise continue the conversation flow
    assistant_reply = next_response['message']
    if 'options' in next_response:
        options_text = "\n\n" + "\n".join([f"• {option}" for option in next_response['options']])
        assistant_reply += options_text

    threads[thread_id].append({"role": "assistant", "content": assistant_reply})

    return jsonify({
        'response': assistant_reply,
        'thread_id': thread_id,
        'conversation_state': get_conversation_state(thread_id)
    })

@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    data = request.get_json()
    thread_id = data.get('thread_id')
    if thread_id:
        threads.pop(thread_id, None)
        conversation_states.pop(thread_id, None)
    return jsonify({'status': 'reset', 'message': 'Conversation reset successfully'})

if __name__ == '__main__':
    app.run(debug=True)
