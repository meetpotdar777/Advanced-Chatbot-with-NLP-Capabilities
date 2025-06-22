🤖 Advanced Chatbot with NLP Capabilities

Project Overview

This project implements a console-based chatbot significantly enhanced with Natural Language Processing (NLP) capabilities, including integration with the Google Gemini API for generative AI responses. It's designed to understand user intentions, extract key information (entities), and provide relevant responses based on both a predefined set of conversational flows and a large language model.

This system demonstrates a more sophisticated approach to building interactive conversational agents by combining traditional NLP techniques with the power of modern generative AI, allowing for more dynamic and intelligent interactions.

✨ Features

Intelligent Intent Recognition: Uses a TF-IDF vectorizer and a Logistic Regression classifier to determine the user's underlying intention (e.g., greeting, product inquiry, order status). Expanded patterns for robust recognition of diverse inputs.

Gemini API Integration: Leverages the Gemini 2.0 Flash model to generate responses for general inquiries or when the chatbot's internal intent recognition has low confidence. This provides a vast knowledge base and conversational flexibility.

Contextual Responses: Provides varied and relevant responses based on the identified intent, and dynamically generates responses for broad questions via Gemini.

Simple Entity Extraction: Demonstrates how to extract specific pieces of information (e.g., an order number) using regular expressions.

New "Advanced Skills" (Intents):

current_time: Provides the current system time.

weather_inquiry: Simulates providing weather information (acknowledges the request).

define_word: Conceptually acknowledges requests for definitions.

joke: Tells simple jokes.

how_are_you: Responds to common small talk about the bot's well-being.

about_bot / capabilities: Explains what the chatbot is and what it can do.

feedback: Guides users on how to provide feedback.

good_bad: Handles general positive/negative acknowledgments.

Robust Preprocessing: Cleans and preprocesses input text for NLP, including lowercasing, tokenization, removing punctuation, and lemmatization. Preprocessing is tuned to preserve important short words for intent matching.

Confidence Scoring & Fallback: Predicts intent with a confidence score. If confidence is low, it intelligently routes the query to the Gemini API (general_inquiry intent) or provides a generic fallback message.

Command-Line Interface: A user-friendly console application for easy interaction.

🧠 How It Works (Technical Flow)

Initialization (__init__):

Downloads necessary NLTK data (stopwords, WordNet, Punkt tokenizer models).

Loads a comprehensive dictionary of predefined intents, each containing patterns (example phrases) and responses.

Trains a scikit-learn Pipeline (intent_pipeline) for intent classification.

Text Preprocessing (_preprocess_text):

Converts text to lowercase, tokenizes it into words.

Removes non-alphabetic characters.

Applies lemmatization (reducing words to their base form) to standardize words. Stopword removal is avoided for initial intent matching to preserve meaning in short phrases.

Intent Model Training (_train_intent_model):

Collects all patterns from the intents dictionary and preprocesses them.

Uses a scikit-learn Pipeline with:

TfidfVectorizer: Converts preprocessed text into numerical TF-IDF features, weighting words by their importance.

LogisticRegression: A classification model that learns to map TF-IDF features to specific intents.

The pipeline is trained on the prepared patterns and their corresponding intent labels.

Intent Recognition (_recognize_intent):

Preprocesses the user's input.

Uses the trained intent_pipeline to predict the most likely intent.

Calculates a confidence score for the prediction.

If the confidence is below a set threshold (e.g., 0.3), the input is classified as "general_inquiry" (to be handled by Gemini) or "no_intent" if the input is empty.

Entity Extraction (_extract_entities):

A simplified function that uses regular expressions (re) to extract specific information (entities) like an order number if the recognized intent is order_status. For a more advanced system, Named Entity Recognition (NER) models (like spaCy's) would be integrated here.

Gemini API Call (_call_gemini_api):

When the intent is general_inquiry (or low confidence leads to it), this method is invoked.

It sends the user's raw input as a prompt to the Gemini 2.0 Flash model via a requests POST call.

The model generates a relevant textual response, leveraging its broad knowledge base.

Includes error handling for API call failures.

Response Generation (get_response):

Determines the intent and extracts entities.

If the intent is general_inquiry, it calls the Gemini API to get a response.

For other recognized intents, it selects a random predefined response from the intents dictionary.

Dynamically integrates extracted entities (like order numbers) or real-time data (like current time) into the responses where applicable.

🚀 Setup and Installation

To get this chatbot running on your local machine, follow these steps:

Save the Code:

Save the provided Python code into a file named Advanced_Chatbot_With_NLP.py (or any other name) on your computer.

Create and Activate a Virtual Environment (Highly Recommended):

It's best practice to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

Open your terminal or command prompt.

Create a new virtual environment:

python -m venv chatbot_env

Activate the environment:

On Windows:

.\chatbot_env\Scripts\activate

On macOS/Linux:

source chatbot_env/bin/activate

Your prompt should change to (chatbot_env) indicating the environment is active.

Install Required Libraries:

With your chatbot_env activated, install the necessary Python packages:

pip install nltk scikit-learn numpy requests transformers

nltk: Natural Language Toolkit for text processing.

scikit-learn: For TF-IDF vectorization and Logistic Regression.

numpy: Fundamental package for numerical computation.

requests: For making HTTP requests to the Gemini API.

transformers: Hugging Face's library for state-of-the-art NLP models (though explicitly used only for import pipeline here, it prepares for future advanced integration).

Download NLTK Data:

The script will automatically attempt to download necessary NLTK data (stopwords, wordnet, omw-1.4, punkt, punkt_tab, averaged_perceptron_tagger) the first time it runs. Ensure you have an active internet connection when you run the script for the very first time for these downloads to succeed. You will see download messages in your console.

PUT YOUR GOOGLE API KEY HERE:

api_key = "AIzaSyAGZQsMu3Cazmu-aj2Y2jzGKZZ8ElPripc"
api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

Run the Chatbot Script:

Once all installations and downloads are complete, run your chatbot script while the chatbot_env is active:

python Advanced_Chatbot_With_NLP.py

🎮 Usage

The chatbot will initialize and then present a prompt for your input.

Chatbot initialized and ready!

Type 'bye', 'quit', or 'exit' to end the conversation.
Try asking about products, your order, or general questions like 'What is AI?' or 'Tell me about the capital of France.'
You:

Interact: Type your messages at the You: prompt.

Exit: Type bye, quit, exit, or similar farewell phrases to end the conversation.

Examples of Interactions:

You: Hi
Bot: Hello! How can I assist you?

You: What products do you offer?
Bot: We offer a variety of products including electronics, clothing, and home goods.

You: What time is it?
Bot: The current time is 10:30:45 AM GMT+5:30. (Time will vary)

You: Tell me a joke.
Bot: Why don't scientists trust atoms? Because they make up everything!

You: Where is my order ID-12345?
Bot: Checking status for order number: ID-12345. Please wait a moment.

You: What is the capital of France?
Bot: (Thinking with Gemini...)
Bot: The capital of France is Paris.

You: Explain quantum physics.
Bot: (Thinking with Gemini...)
Bot: Quantum physics is the study of matter and energy at the most fundamental level. It aims to describe the properties and behavior of the building blocks of nature, such as atoms and subatomic particles. Unlike classical physics, which describes the macroscopic world, quantum physics operates on principles that often seem counter-intuitive, like superposition (particles existing in multiple states simultaneously) and entanglement (particles being linked regardless of distance).

You: Thank you
Bot: You're welcome!

You: Bye
Bot: Goodbye! Have a great day!

⚠️ Limitations

While significantly improved, this project remains a demonstration and has limitations for production use:

Rule-Based Entity Extraction: Uses simple regular expressions for entity extraction, which is brittle and not robust to variations. Real systems use sophisticated Named Entity Recognition (NER) models (e.g., from spaCy).

Limited Context Management: The chatbot is largely "stateless" beyond the current turn. It doesn't inherently remember conversation history for multi-turn dialogues (e.g., "What about product X?" after an initial product inquiry).

Predefined Intents for Specific Tasks: While Gemini handles general knowledge, specific operational tasks (like checking order status) still rely on the explicitly defined intents and patterns.

Gemini API Rate Limits: Excessive use of the Gemini API might hit rate limits in a production setting, requiring proper API key management and potentially paid tiers.

No Database Integration: Intents and data are hardcoded. A real application would manage these in a database for scalability.

No Multi-lingual Support: Currently, only English is supported.

Error Handling (Gemini): Basic error handling for API calls. More robust retry mechanisms and detailed logging would be needed in production.

🔮 Future Enhancements

Full spaCy Integration: Uncomment and utilize self.nlp = spacy.load("en_core_web_sm") to perform advanced linguistic analysis (POS tagging, NER, dependency parsing) on user input for more precise entity extraction and intent understanding.

Contextual Dialogue Management: Implement a state machine or a more advanced dialogue management framework (e.g., using a library like Rasa) to manage conversation flow across multiple turns.

Custom Models with Transformers: For specific, complex domains, fine-tune pre-trained transformers models (e.g., BERT, RoBERTa) on your own custom datasets for highly accurate intent recognition and entity extraction, replacing or augmenting the current TF-IDF + Logistic Regression approach.

Knowledge Base Integration: Connect the chatbot to a structured knowledge base (e.g., a database of products, FAQs) to retrieve specific information more efficiently than relying solely on a generative model for predefined questions.

Sentiment Analysis: Integrate a sentiment analysis model (e.g., from NLTK's VADER, TextBlob, or a transformer-based model) to understand the user's emotional tone and tailor responses.

Speech-to-Text (STT) and Text-to-Speech (TTS): Integrate STT for voice input and TTS for voice output to create a voice-enabled assistant.

Web User Interface: Develop a full web application (e.g., using Flask, Django, Streamlit, or React) for an intuitive graphical interface.

Persistent Chat History: Implement a system to store conversation history (e.g., in a local file or a database) for a more continuous user experience.
