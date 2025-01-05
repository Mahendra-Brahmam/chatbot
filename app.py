# pip install nltk scikit-learn streamlit

import os
import json
import nltk
import ssl
import streamlit as st
import random
import csv
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Configure SSL for downloading NLTK data
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Load intents from JSON file
file_path = os.path.abspath("intents.json")
with open("D:\python\intents.json", 'r') as file:
    Intents = json.load(file)

# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=1000)

# Prepare training data
tags = []
patterns = []
for intent in Intents['intents']:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    print(f"Predicted tag: {tag}")
    for intent in Intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            print(f"Response: {response}")
            return response
    return "I'm sorry, I didn't understand that."


import os
import csv
import datetime
import streamlit as st
from PIL import Image

# Initialize a counter
counter = 0

def main():
    global counter

    # Set page config for better UI
    st.set_page_config(
        page_title="Chatbot with NLP",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Add a background color
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .title {
            color: #4CAF50;
            font-size: 3rem;
            font-weight: bold;
        }
        .header {
            color: #333;
            font-size: 1.5rem;
            font-weight: bold;
        }
        .subheader {
            color: #666;
            font-size: 1.2rem;
        }
        .frame {
            border: 2px solid #4CAF50;
            padding: 15px;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Add a sidebar image
    sidebar_image = Image.open("D:\python\Edunet Foundation named chatbot logo with vibrant colors.png")  # Replace with your image file
    st.sidebar.image(sidebar_image, use_container_width=True)

    st.sidebar.title("Navigation")

    # Create a sidebar menu with options
    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.radio("Menu", menu)

    # Home Menu
    if choice == "Home":
        st.markdown("<div class='title'>Welcome to the Chatbot</div>", unsafe_allow_html=True)
        st.markdown(
            """<div class='subheader'>Start chatting below. Enter a message and press Enter to begin!</div>""",
            unsafe_allow_html=True,
        )

        user_input = st.text_input("You:", key=f"user_input_{counter}", placeholder="Type your message here...")

        if user_input:
            # Increment the counter for unique keys
            counter += 1

            # Generate chatbot response
            response = chatbot(user_input)

            # Display chatbot response inside a styled frame
            st.markdown(f"<div class='frame'><strong>Chatbot:</strong> {response}</div>", unsafe_allow_html=True)

            # Get the current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save the conversation to a CSV file
            if not os.path.exists('chat_log.csv'):
                with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            # Check for a goodbye message
            if response.lower() in ['goodbye', 'bye']:
                st.success("Thank you for chatting with me. Have a great day!")
                st.stop()

    # Conversation History Menu
    elif choice == "Conversation History":
        st.markdown("<div class='header'>Conversation History</div>", unsafe_allow_html=True)
        try:
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    st.markdown(f"<div class='frame'>User: {row[0]}<br>Chatbot: {row[1]}<br>Timestamp: {row[2]}</div>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.info("No conversation history available.")

    # About Menu
    elif choice == "About":
        st.markdown("<div class='header'>About</div>", unsafe_allow_html=True)
        st.image("D:\python\chatbot displaying text about.png", width=400)  # Replace with your image file
        st.markdown(
            """
            <div class='subheader'>
            The goal of this project is to create a chatbot that can understand and respond to user intents using NLP techniques.
            </div>
            <ul>
                <li><strong>Training:</strong> The chatbot uses Logistic Regression and NLP techniques for intent recognition.</li>
                <li><strong>Interface:</strong> Built using Streamlit for an interactive user experience.</li>
                <li><strong>Dataset:</strong> A labeled dataset of intents and entities.</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

# Example chatbot function for demonstration
# Replace this with your actual chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    print(f"Predicted tag: {tag}")
    for intent in Intents['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            print(f"Response: {response}")
            return response
    return "I'm sorry, I didn't understand that."


# Run the application
if __name__ == '__main__':
    main()
