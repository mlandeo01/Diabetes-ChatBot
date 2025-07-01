from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import logging
from openai import OpenAI
from uuid import uuid4
from datetime import datetime, timedelta
import re
import json

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enhanced in-memory storage
threads = {}
conversation_states = {}
user_profiles = {}
blood_sugar_logs = {}

ENHANCED_SYSTEM_PROMPT = """
You are DiaFriend, a compassionate and knowledgeable diabetes management assistant designed to support people living with diabetes.

**Your Core Mission:**
- Provide emotional support and practical guidance for daily diabetes management
- Help users track and understand their blood glucose patterns
- Offer evidence-based information about diabetes care
- Encourage healthy lifestyle choices and medication adherence
- Create a safe, non-judgmental space for diabetes-related concerns

**Key Capabilities:**
1. **Blood Sugar Management**: Help log readings, identify patterns, and understand target ranges
2. **Nutrition Guidance**: Provide carb counting tips, meal planning suggestions, and food impact insights
3. **Exercise Support**: Offer safe exercise recommendations and blood sugar monitoring advice
4. **Medication Reminders**: Help with timing and adherence strategies
5. **Emotional Support**: Address diabetes burnout, anxiety, and daily challenges
6. **Emergency Recognition**: Identify urgent situations requiring immediate medical attention

**Communication Style:**
- Use warm, empathetic, and encouraging language
- Avoid medical jargon; explain terms when necessary
- Ask thoughtful follow-up questions to better understand context
- Acknowledge the emotional aspects of diabetes management
- Celebrate small victories and progress
- Normalize the challenges of living with diabetes

**Safety Guidelines:**
- NEVER diagnose medical conditions or replace professional medical advice
- Always recommend consulting healthcare providers for:
  * Persistent high or low blood sugars
  * New symptoms or concerns
  * Medication adjustments
  * Emergency situations (severe highs/lows, DKA symptoms)
- Encourage regular medical check-ups and lab work
- Promote evidence-based diabetes care practices

**Response Framework:**
1. Acknowledge the user's concern or situation
2. Provide relevant, accurate information
3. Offer practical next steps or suggestions
4. Include emotional support when appropriate
5. End with an open invitation for follow-up questions

**Emergency Indicators to Watch For:**
- Blood glucose >400 mg/dL or <70 mg/dL with symptoms
- Signs of DKA (nausea, vomiting, fruity breath, confusion)
- Severe hypoglycemia symptoms
- Persistent illness affecting blood sugar control

Remember: You're not just providing information—you're being a supportive companion in their diabetes journey.
"""

def get_conversation_state(thread_id):
    """Get current conversation state with enhanced tracking."""
    default_state = {
        'step': 'greeting',
        'context': 'general',
        'last_bg_reading': None,
        'tracking_goal': None,
        'user_data': {},
        'conversation_history': [],
        'needs_medical_attention': False
    }
    return conversation_states.get(thread_id, default_state)

def update_conversation_state(thread_id, updates):
    """Update conversation state with better data persistence."""
    if thread_id not in conversation_states:
        conversation_states[thread_id] = {
            'step': 'greeting',
            'context': 'general',
            'last_bg_reading': None,
            'tracking_goal': None,
            'user_data': {},
            'conversation_history': [],
            'needs_medical_attention': False
        }
    conversation_states[thread_id].update(updates)

def validate_blood_sugar(reading_str):
    """Validate and parse blood sugar reading."""
    try:
        # Extract numbers from string
        numbers = re.findall(r'\d+', reading_str)
        if not numbers:
            return None, "Please provide a numeric blood sugar reading.", None
        
        reading = int(numbers[0])
        
        # Validate reasonable range
        if reading < 20 or reading > 600:
            return None, "That reading seems unusually high or low. Please double-check and enter again, or contact your healthcare provider if accurate.", None
        
        # Provide context based on reading
        if reading < 70:
            context = "⚠️ This is a low reading. If you're experiencing symptoms, treat immediately with fast-acting carbs."
            urgent = True
        elif reading > 250:
            context = "⚠️ This is a high reading. Stay hydrated and monitor for ketones if you have Type 1 diabetes."
            urgent = True
        elif reading >= 70 and reading <= 180:
            context = "✅ This reading is within a good range for most people."
            urgent = False
        else:
            context = "This reading is elevated but manageable. Consider factors like recent meals or stress."
            urgent = False
            
        return reading, context, urgent
        
    except ValueError:
        return None, "Please enter a valid number for your blood sugar reading.", None

def analyze_bg_pattern(user_id, new_reading):
    """Analyze blood sugar patterns for insights."""
    if user_id not in blood_sugar_logs:
        blood_sugar_logs[user_id] = []
    
    # Add new reading with timestamp
    blood_sugar_logs[user_id].append({
        'reading': new_reading,
        'timestamp': datetime.now(),
        'date': datetime.now().strftime('%Y-%m-%d')
    })
    
    # Keep only last 30 readings
    blood_sugar_logs[user_id] = blood_sugar_logs[user_id][-30:]
    
    readings = blood_sugar_logs[user_id]
    if len(readings) < 2:
        return "Great start on tracking! Keep logging readings to identify patterns."
    
    # Basic pattern analysis
    recent_readings = [r['reading'] for r in readings[-7:]]  # Last week
    avg_recent = sum(recent_readings) / len(recent_readings)
    
    if len(readings) >= 7:
        trend_analysis = []
        if avg_recent > 180:
            trend_analysis.append("Recent readings are running high. Consider discussing meal timing and portions with your healthcare team.")
        elif avg_recent < 100:
            trend_analysis.append("Great control! Your recent readings show good management.")
        
        # Check for variability
        if max(recent_readings) - min(recent_readings) > 100:
            trend_analysis.append("Your readings show some variability. Consistent meal timing and stress management can help.")
        
        return " ".join(trend_analysis) if trend_analysis else "Your readings show you're working hard at management. Keep it up!"
    
    return "Keep tracking! More data will help identify helpful patterns."

def get_contextual_response(state, user_message):
    """Generate contextual responses based on conversation state."""
    step = state['step']
    context = state['context']
    
    # Handle blood sugar logging
    if 'blood sugar' in user_message.lower() or 'glucose' in user_message.lower() or any(char.isdigit() for char in user_message):
        if re.search(r'\d+', user_message):
            reading, message, urgent = validate_blood_sugar(user_message)
            if reading:
                thread_id = state.get('thread_id', 'default')
                pattern_insight = analyze_bg_pattern(thread_id, reading)
                
                return {
                    'response': f"Thanks for logging {reading} mg/dL. {message}\n\n📊 Pattern insight: {pattern_insight}\n\nWhat else can I help you with today?",
                    'context': 'logged_reading',
                    'needs_medical_attention': urgent,
                    'quick_actions': ['Log another reading', 'Ask about this reading', 'Nutrition tips', 'Exercise guidance']
                }
            else:
                return {
                    'response': message,
                    'context': 'clarification_needed',
                    'quick_actions': ['Try logging again', 'Ask a different question']
                }
    
    # Handle different conversation contexts
    if step == 'greeting':
        return {
            'response': "Hello! I'm DiaFriend, your diabetes management companion. 💙\n\nI'm here to help with blood sugar tracking, nutrition questions, exercise tips, or just to listen if you need support with diabetes challenges.\n\nWhat's on your mind today?",
            'context': 'general',
            'quick_actions': ['Log blood sugar', 'Nutrition help', 'Exercise tips', 'Emotional support', 'General question']
        }
    
    # Handle different topics
    user_lower = user_message.lower()
    
    if any(word in user_lower for word in ['carb', 'food', 'eat', 'meal', 'diet', 'nutrition']):
        return {
            'response': "I'd love to help with nutrition! Nutrition is such a key part of diabetes management, and it can feel overwhelming sometimes.",
            'context': 'nutrition',
            'generate_ai_response': True
        }
    
    elif any(word in user_lower for word in ['exercise', 'workout', 'activity', 'gym', 'walk']):
        return {
            'response': "Exercise is wonderful for blood sugar management! Let me help you with safe and effective approaches.",
            'context': 'exercise',
            'generate_ai_response': True
        }
    
    elif any(word in user_lower for word in ['stress', 'worry', 'overwhelm', 'burn', 'tired', 'frustrated']):
        return {
            'response': "I hear you - diabetes can be emotionally challenging, and those feelings are completely valid.",
            'context': 'emotional_support',
            'generate_ai_response': True
        }
    
    else:
        return {
            'response': "I'm here to help with your diabetes question.",
            'context': 'general',
            'generate_ai_response': True
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        thread_id = data.get('thread_id')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Initialize or get existing thread
        if not thread_id or thread_id not in threads:
            thread_id = str(uuid4())
            threads[thread_id] = [{"role": "system", "content": ENHANCED_SYSTEM_PROMPT}]

        # Get current state and generate contextual response
        state = get_conversation_state(thread_id)
        state['thread_id'] = thread_id
        
        contextual_response = get_contextual_response(state, message)
        
        # Add user message to thread
        threads[thread_id].append({"role": "user", "content": message})
        
        # Handle AI-generated responses
        if contextual_response.get('generate_ai_response', False):
            try:
                # Create context-aware prompt
                context_prompt = f"""
The user just said: "{message}"

Context: {contextual_response['context']}

Please provide a helpful, empathetic response following your role as DiaFriend. 
Keep it conversational (2-3 paragraphs max) and include:
1. Acknowledgment of their situation
2. Practical, actionable advice
3. Encouragement
4. An invitation for follow-up questions

Remember your safety guidelines - recommend medical consultation when appropriate.
"""

                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=threads[thread_id] + [{"role": "user", "content": context_prompt}],
                    temperature=0.4,
                    max_tokens=400
                )

                ai_response = response.choices[0].message.content
                threads[thread_id].append({"role": "assistant", "content": ai_response})
                
                # Update conversation state
                update_conversation_state(thread_id, {
                    'step': 'conversation',
                    'context': contextual_response['context'],
                    'last_interaction': datetime.now().isoformat()
                })

                return jsonify({
                    'response': ai_response,
                    'thread_id': thread_id,
                    'quick_actions': contextual_response.get('quick_actions', []),
                    'context': contextual_response['context']
                })

            except Exception as e:
                logger.error(f"Error generating AI response: {e}")
                return jsonify({
                    'response': "I'm having trouble processing that right now. Could you try rephrasing your question?",
                    'thread_id': thread_id,
                    'error': 'ai_generation_failed'
                }), 500

        # Handle direct contextual responses
        else:
            assistant_reply = contextual_response['response']
            threads[thread_id].append({"role": "assistant", "content": assistant_reply})
            
            # Update conversation state
            update_conversation_state(thread_id, {
                'step': 'conversation',
                'context': contextual_response['context'],
                'needs_medical_attention': contextual_response.get('needs_medical_attention', False),
                'last_interaction': datetime.now().isoformat()
            })

            return jsonify({
                'response': assistant_reply,
                'thread_id': thread_id,
                'quick_actions': contextual_response.get('quick_actions', []),
                'context': contextual_response['context'],
                'medical_attention': contextual_response.get('needs_medical_attention', False)
            })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': 'An error occurred processing your message',
            'response': "I'm sorry, I'm having technical difficulties. Please try again in a moment."
        }), 500

@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    """Reset conversation with better cleanup."""
    try:
        data = request.get_json()
        thread_id = data.get('thread_id')
        
        if thread_id:
            threads.pop(thread_id, None)
            conversation_states.pop(thread_id, None)
            # Keep blood sugar logs for continuity
            logger.info(f"Reset conversation for thread: {thread_id}")
        
        return jsonify({
            'status': 'success',
            'message': 'Conversation reset successfully. Your blood sugar logs are preserved.'
        })
    
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        return jsonify({'error': 'Failed to reset conversation'}), 500

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get user's blood sugar statistics."""
    try:
        thread_id = request.args.get('thread_id')
        if not thread_id or thread_id not in blood_sugar_logs:
            return jsonify({'message': 'No data available yet'})
        
        readings = blood_sugar_logs[thread_id]
        if not readings:
            return jsonify({'message': 'No readings logged yet'})
        
        # Calculate basic stats
        values = [r['reading'] for r in readings]
        stats = {
            'total_readings': len(values),
            'average': round(sum(values) / len(values), 1),
            'highest': max(values),
            'lowest': min(values),
            'in_range': len([v for v in values if 70 <= v <= 180]),
            'recent_trend': 'stable'  # Could be enhanced with actual trend calculation
        }
        
        return jsonify({
            'stats': stats,
            'recent_readings': readings[-7:]  # Last 7 readings
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Failed to retrieve statistics'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)