import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import random
import re
import datetime # Added for current_time intent
import requests # Added for making HTTP requests to Gemini API
import json # Added for handling JSON payloads with Gemini API
import spacy # Added for more advanced NLP capabilities (conceptual integration for now)
from transformers import pipeline # Uncomment if you plan to use Hugging Face Transformers

# Download necessary NLTK data (if not already downloaded)
# These downloads require an internet connection on the first run.
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4') # Open Multilingual Wordnet
except LookupError:
    nltk.download('omw-1.4')
try:
    nltk.data.find('tokenizers/punkt') # For word_tokenize and sent_tokenize
except LookupError:
    nltk.download('punkt')
try:
    # Explicitly download 'punkt_tab' as suggested by NLTK's traceback for certain versions/languages
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger') # For POS tagging, useful for more advanced NLP
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class Chatbot:
    def __init__(self):
        """
        Initializes the Chatbot with NLP components and trains its intent recognition model.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.intents = self._load_intents()
        self.intent_pipeline = self._train_intent_model()
        
        # Initialize spaCy (uncomment if you've installed 'en_core_web_sm' model)
        # try:
        #     self.nlp = spacy.load("en_core_web_sm")
        #     print("spaCy model loaded.")
        # except OSError:
        #     print("spaCy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
        #     self.nlp = None

        print("Chatbot initialized and ready!")

    def _load_intents(self):
        """
        Loads predefined intents and their associated patterns and responses.
        Expanded patterns for better recognition and added new intents for advanced skills.
        """
        return {
            "greeting": {
                "patterns": [
                    "hi", "hello", "hey", "good morning", "good evening", "greetings",
                    "hello there", "good day", "what's up", "yo", "howdy", "hi bot",
                    "hallo", "good afternoon", "nice to meet you", "how are you today",
                    "are you there", "anyone home", "can you hear me", "hi chat",
                    "hello chatbot", "what's good", "greetings program"
                ],
                "responses": [
                    "Hello!", "Hi there!", "Greetings!", "Hey, how can I help you?",
                    "Hi, nice to see you!", "What can I do for you today?",
                    "Hello! How can I assist you?", "Hi, I'm here to help.",
                    "Hello! It's a pleasure to assist you."
                ]
            },
            "farewell": {
                "patterns": [
                    "bye", "goodbye", "see you", "farewell", "cya", "see ya",
                    "talk to you soon", "until next time", "bye bye", "later",
                    "exit", "quit", "end chat", "finish", "good night", "gotta go",
                    "i'm leaving", "i have to go", "adios", "cheerio"
                ],
                "responses": [
                    "Goodbye!", "See you later!", "Bye for now!", "Have a great day!",
                    "Farewell!", "It was nice chatting with you.", "Talk to you soon!",
                    "See you around!", "Wishing you a great day!"
                ]
            },
            "thanks": {
                "patterns": [
                    "thanks", "thank you", "much appreciated", "cheers", "thank you so much",
                    "many thanks", "appreciate it", "thx", "thank you for your help",
                    "thanks a lot", "you're a lifesaver", "gracias", "thank you very much"
                ],
                "responses": [
                    "You're welcome!", "No problem!", "Glad to help!", "Anytime!",
                    "My pleasure!", "You got it!", "Happy to assist!", "Don't mention it!"
                ]
            },
            "product_inquiry": {
                "patterns": [
                    "what products do you offer", "tell me about your services", "what can I buy",
                    "product list", "do you sell anything", "what services do you provide",
                    "products you have", "what do you sell", "show me your items",
                    "list of services", "types of products", "catalog", "do you have"
                ],
                "responses": [
                    "We offer a variety of products including electronics, clothing, and home goods.", 
                    "Our services range from tech support to personalized shopping assistance. What are you looking for?",
                    "We specialize in consumer electronics, fashion, and home decor. Is there a specific category you're interested in?",
                    "Our product catalog includes gadgets, apparel, and household items. What are you curious about?"
                ]
            },
            "order_status": {
                "patterns": [
                    "where is my order", "order status", "track my package", "delivery update",
                    "check my order", "my package status", "is my order shipped",
                    "when will my order arrive", "status of my delivery", "track order",
                    "what's up with my delivery", "find my order"
                ],
                "responses": [
                    "Please provide your order number.", "Could you give me your order ID so I can check for you?",
                    "I can help with that. What is your order number?",
                    "To track your order, please give me the order ID."
                ]
            },
            "contact_support": {
                "patterns": [
                    "contact support", "help me", "I need assistance", "talk to a human",
                    "customer service", "can I speak to someone", "live agent", "get human help",
                    "support", "reach customer service", "need to talk to someone",
                    "speak to representative", "human support"
                ],
                "responses": [
                    "You can reach our support team at support@example.com or call us at 1-800-123-4567.", 
                    "I can connect you to a human agent if you'd like. Please hold while I transfer you.",
                    "If you need immediate assistance, please call our support line. Otherwise, I can try to help you here.",
                    "For direct human support, our team is available via email or phone."
                ]
            },
            "return_policy": {
                "patterns": [
                    "return policy", "can I return this", "how to return an item", "refund policy",
                    "returns", "exchange policy", "how do I get a refund", "return an item",
                    "what is your return policy", "can I get my money back", "how to exchange"
                ],
                "responses": [
                    "Our return policy allows returns within 30 days of purchase, provided the item is unused and in its original packaging.",
                    "You can find detailed information about returns on our website under the 'Returns' section.",
                    "For returns or exchanges, please visit our 'Returns & Refunds' page on the website.",
                    "Items can be returned within 30 days. Full details are on our website."
                ]
            },
            "payment_methods": {
                "patterns": [
                    "payment methods", "how can I pay", "what payment options do you accept",
                    "credit card", "do you take paypal", "payment options", "how to pay",
                    "can I pay with visa", "accepted cards"
                ],
                "responses": [
                    "We accept Visa, MasterCard, American Express, PayPal, and bank transfers.",
                    "You can pay using major credit cards or PayPal.",
                    "All major credit cards and PayPal are accepted.",
                    "We support several payment methods including credit/debit cards and PayPal."
                ]
            },
            "hours_operation": {
                "patterns": [
                    "what are your hours", "when are you open", "opening times",
                    "business hours", "are you open now", "what time do you open/close",
                    "your operating hours", "when are you available"
                ],
                "responses": [
                    "Our online store is open 24/7! Customer support is available from 9 AM to 6 PM (local time), Monday to Friday.",
                    "Our digital services are always available. For live support, we operate from 9 AM to 6 PM, Monday through Friday.",
                    "You can shop anytime online. Our support team is available during standard business hours."
                ]
            },
            "agent_transfer_confirm": {
                "patterns": [
                    "yes connect me", "transfer me", "yes please", "connect me now",
                    "please connect", "transfer to human", "I want to talk to someone",
                    "go to agent", "put me through to a person"
                ],
                "responses": [
                    "Connecting you to an agent now. Please wait.",
                    "One moment, transferring you to a human representative.",
                    "Alright, I'm transferring you to a support agent."
                ]
            },
            # --- New Advanced Skills / Intents ---
            "about_bot": {
                "patterns": [
                    "who are you", "what are you", "tell me about yourself", "your name",
                    "are you a robot", "what is your purpose", "what kind of bot are you",
                    "your identity", "what do you do"
                ],
                "responses": [
                    "I am a chatbot designed to assist you with common inquiries.",
                    "I am a virtual assistant, here to help.",
                    "You can call me your friendly AI assistant!",
                    "I am an AI-powered conversational agent."
                ]
            },
            "capabilities": {
                "patterns": [
                    "what can you do", "your functions", "how can you help me",
                    "what are your capabilities", "can you help me with", "what services can you offer",
                    "what are your features", "list your features"
                ],
                "responses": [
                    "I can answer questions about our products, services, policies, and more.",
                    "I can help you with order status, returns, payment methods, and general information.",
                    "My capabilities include providing information and directing you to the right resources.",
                    "I'm here to provide information and assist with common customer service queries."
                ]
            },
            "feedback": {
                "patterns": [
                    "I have feedback", "give feedback", "suggestions", "I want to give feedback",
                    "how can I give feedback", "comment", "suggestions for improvement",
                    "send feedback", "feedback please"
                ],
                "responses": [
                    "Thank you for your feedback! Please send your suggestions to feedback@example.com.",
                    "We appreciate your input. What kind of feedback do you have?",
                    "Your feedback helps us improve. Please share your thoughts.",
                    "We'd love to hear your feedback. You can email us or tell me now."
                ]
            },
            "how_are_you": { # Adding a common small talk intent
                "patterns": [
                    "how are you", "how are you doing", "are you fine", "how have you been",
                    "how's life", "all good", "you alright", "how are you feeling"
                ],
                "responses": [
                    "I'm doing great, thank you for asking!",
                    "I'm functioning perfectly, thanks!",
                    "As an AI, I don't have feelings, but I'm ready to assist you!",
                    "I am an AI, so I don't experience 'how I am', but I'm ready to help!"
                ]
            },
            "current_time": {
                "patterns": [
                    "what time is it", "current time", "time now", "tell me the time",
                    "what's the time", "show me the time"
                ],
                "responses": [
                    "The current time is {time}.",
                    "It's currently {time}.",
                    "Right now, it's {time}."
                ]
            },
            "weather_inquiry": { # Simulated weather intent
                "patterns": [
                    "what's the weather like", "weather forecast", "how's the weather",
                    "will it rain today", "weather in city", "temperature",
                    "current weather", "weather today"
                ],
                "responses": [
                    "I can't check live weather, but I can tell you that weather information is usually available on dedicated weather apps or websites.",
                    "Unfortunately, I don't have access to real-time weather data.",
                    "For weather updates, please check your local weather service.",
                    "My apologies, I am not connected to a weather service at the moment."
                ]
            },
            "define_word": { # Simulated definition intent
                "patterns": [
                    "define", "what does mean", "meaning of", "what is a",
                    "definition of", "what does x mean" # Added x for generic example
                ],
                "responses": [
                    "I can't look up dictionary definitions directly, but I can try to help with other queries.",
                    "While I don't have a built-in dictionary, you can often find definitions online very quickly.",
                    "Could you ask me something related to our products or services instead?",
                    "I am not a dictionary, but I can assist with company-specific information."
                ]
            },
            "joke": { # Simple joke intent
                "patterns": [
                    "tell me a joke", "make me laugh", "joke", "funny story",
                    "tell a joke", "got any jokes"
                ],
                "responses": [
                    "Why don't scientists trust atoms? Because they make up everything!",
                    "Did you hear about the mathematician who was afraid of negative numbers? He'd stop at nothing to avoid them!",
                    "Why was the computer cold? It left its Windows open!",
                    "What do you call a fake noodle? An impasta!"
                ]
            },
             "good_bad": { # Intent for general positive/negative statements / acknowledgements
                "patterns": [
                    "ok", "okay", "alright", "good", "great", "nice", "awesome", "perfect",
                    "bad", "not good", "terrible", "horrible", "wrong", "understood", "got it"
                ],
                "responses": [
                    "Okay, how can I proceed to help you?",
                    "Alright, what's next?",
                    "Understood. How may I assist further?",
                    "I see. How can I help you from here?",
                    "Thanks for letting me know."
                ]
            },
            "general_inquiry": { # Broad category for queries that might go to Gemini
                "patterns": [
                    "tell me about", "what is", "how do i", "explain", "information about",
                    "can you explain", "what's the deal with"
                ],
                "responses": [""] # Responses will be handled by Gemini API
            }
        }

    def _preprocess_text(self, text):
        """
        Cleans and preprocesses input text for NLP.
        Steps: lowercase, tokenize, remove punctuation, lemmatize.
        Stopword removal is *not* performed here to retain important short words
        for intent matching, especially in basic greetings/commands.
        """
        text = text.lower()
        words = nltk.word_tokenize(text)
        # Filter out punctuation and non-alphabetic tokens.
        # For more advanced preprocessing (e.g., entity recognition),
        # a spaCy pipeline (self.nlp) would be used here.
        words = [self.lemmatizer.lemmatize(word) for word in words if word.isalpha()]
        return ' '.join(words)

    def _train_intent_model(self):
        """
        Trains a classification model to identify the user's intent.
        Uses TF-IDF for feature extraction and Logistic Regression for classification.
        """
        patterns = []
        labels = []
        for intent, data in self.intents.items():
            # Exclude "general_inquiry" patterns from direct training if they are empty
            # and meant only for Gemini fallback. If they had actual patterns, they'd be included.
            if intent == "general_inquiry" and not data["patterns"]:
                continue
            for pattern in data["patterns"]:
                patterns.append(self._preprocess_text(pattern))
                labels.append(intent)

        # Create a pipeline: TF-IDF Vectorizer + Logistic Regression Classifier
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LogisticRegression(random_state=42, max_iter=2000)) # Increased max_iter for convergence
        ])
        
        # Ensure there are patterns to train on
        if not patterns:
            print("Warning: No patterns found for intent training. Intent recognition may not work.")
            return None # Or raise an error as appropriate

        pipeline.fit(patterns, labels)
        print("Intent model trained successfully.")
        return pipeline

    def _recognize_intent(self, user_input):
        """
        Predicts the intent of the user's input.
        Adjusted confidence threshold for better recognition on a diverse dataset.
        """
        preprocessed_input = self._preprocess_text(user_input)
        if not preprocessed_input.strip(): # Check if it's truly empty after preprocessing
            return "no_intent"
        
        # Predict intent
        predicted_intent = self.intent_pipeline.predict([preprocessed_input])[0]
        
        # Get prediction probabilities for confidence
        probabilities = self.intent_pipeline.predict_proba([preprocessed_input])[0]
        # Get the highest probability
        confidence = np.max(probabilities)

        # Basic confidence threshold (adjusted for more lenient initial recognition, especially with more patterns)
        # Tuning this value is crucial. 0.3 can be a good starting point for a small dataset with many patterns.
        if confidence < 0.3: # Lowered threshold to be more forgiving for simple inputs
            # If low confidence, but not an empty input, fall back to "general_inquiry" to try Gemini
            return "general_inquiry" if preprocessed_input.strip() else "no_intent"
        
        return predicted_intent

    def _extract_entities(self, user_input, intent):
        """
        (Simplified) Extracts entities based on the recognized intent.
        For demonstration, uses regex for a specific example (order number).
        For more advanced entity extraction, consider using spaCy's NER.
        """
        entities = {}
        if intent == "order_status":
            # Example: Extracting a potential order number (e.g., "ORD12345", "order 9876", "1234567")
            # Improved regex to be more flexible
            match = re.search(r'(?:order|id|num|#)?\s*([a-zA-Z]{2,}\d{3,}|\d{5,})', user_input, re.IGNORECASE)
            if match:
                entities["order_number"] = match.group(1).upper() # Store as uppercase for consistency
            else:
                # Fallback to just finding digits if no prefix
                match = re.search(r'(\d{5,})', user_input)
                if match:
                    entities["order_number"] = match.group(1)
        # You could add more entity extraction for other intents here if needed
        # e.g., for 'define_word', extract the word to define.
        return entities

    def _call_gemini_api(self, prompt):
        """
        Calls the Gemini API to generate a response for queries that fall into
        the "general_inquiry" intent or have low confidence.
        """
        # Placeholder for the API key. Canvas will inject it at runtime.
        # DO NOT put your actual API key here.
        api_key = "AIzaSyAGZQsMu3Cazmu-uj2Y2rzGKeZ8ElPripc"
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

        chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
        payload = {"contents": chat_history}

        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(api_url, headers=headers, json=payload)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            result = response.json()

            if result.get("candidates") and len(result["candidates"]) > 0 and \
               result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
               len(result["candidates"][0]["content"]["parts"]) > 0:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return text
            else:
                print("Gemini API: Unexpected response structure or no content.")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Gemini API Error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Gemini API JSON Decode Error: {e}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred with Gemini API: {e}")
            return None

    def get_response(self, user_input):
        """
        Generates a response based on user input, leveraging Gemini API for general inquiries.
        """
        intent = self._recognize_intent(user_input)
        entities = self._extract_entities(user_input, intent)

        if intent == "no_intent":
             return random.choice([
                "How can I assist you today?",
                "Please tell me what you need.",
                "What can I do for you?"
            ])
        elif intent == "general_inquiry":
            print("Bot: (Thinking with Gemini...)") # Inform user about API call
            gemini_response = self._call_gemini_api(user_input)
            if gemini_response:
                return gemini_response
            else:
                # Fallback if Gemini API fails or returns no content
                return random.choice([
                    "I'm sorry, I couldn't process that request at the moment. Can I help with something else?",
                    "It seems I'm having trouble understanding. Please try a different query or rephrase."
                ])
        else: # Recognized intent
            responses = self.intents.get(intent, {}).get("responses", ["I'm not sure how to respond to that."])
            response = random.choice(responses)

            # Integrate entities or dynamic content into responses where applicable
            if intent == "order_status" and "order_number" in entities:
                response = f"Checking status for order number: {entities['order_number']}. Please wait a moment."
            elif intent == "current_time":
                now = datetime.datetime.now()
                # Get current location for accurate time, or fallback to system time
                # For a real application, you might use a geo-IP service for location or ask the user.
                # For this conceptual example, it will use the system's current time.
                response = response.format(time=now.strftime("%I:%M:%S %p %Z").strip()) # Format current time
            
            return response

# Main execution
if __name__ == "__main__":
    chatbot = Chatbot()
    print("\nType 'bye', 'quit', or 'exit' to end the conversation.")
    print("Try asking about products, your order, or general questions like 'What is AI?' or 'Tell me about the capital of France.'")

    while True:
        user_input = input("You: ")
        # Explicitly handle common exit commands before intent recognition for robustness
        if user_input.lower() in ["bye", "quit", "exit", "goodbye", "see you", "good night", "gotta go", "i'm leaving", "i have to go", "adios", "cheerio"]:
            print(f"Bot: {chatbot.get_response('bye')}") # Use farewell intent response
            break
        
        response = chatbot.get_response(user_input)
        print(f"Bot: {response}")
        # Optional: Add a delay to simulate thinking time
