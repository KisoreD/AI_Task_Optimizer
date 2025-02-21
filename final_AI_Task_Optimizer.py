import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace
from textblob import TextBlob
import time
import random
import matplotlib.pyplot as plt
import json
import hashlib
import os
from collections import Counter

# -----------------------------------
# 💡 AI Task Optimizer - Mood Detection & Task Recommendation
# This script detects facial emotions in real-time using DeepFace and analyzes text sentiment with TextBlob.
# It recommends tasks based on detected mood and monitors stress levels in employees.
# -----------------------------------

# 🎭 Function to detect facial emotion from webcam feed
def detect_facial_emotion(frame):
    """
    Analyzes the input video frame and detects the dominant facial emotion.
    Uses DeepFace for emotion recognition.
    Returns: Detected emotion (str)
    """
    result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
    if result:
        return result[0]['dominant_emotion'].capitalize()  # Convert first letter to uppercase
    return "Neutral"

# 📖 Function to analyze text sentiment
def detect_text_emotion(text):
    """
    Analyzes the sentiment polarity of a given text using TextBlob.
    Returns:
      - "Happy" for positive polarity
      - "Sad" for negative polarity
      - "Fear" for neutral polarity
    """
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Happy"
    elif polarity < 0:
        return "Sad"
    return "Fear"

# ⚠️ Function to monitor stress levels based on recent moods
def monitor_stress(mood_history):
    """
    Checks if an employee has experienced a high number of 'Sad' moods in recent entries.
    If stress exceeds a threshold, it triggers an HR alert.
    """
    stress_threshold = 3  # Define threshold for stress alerts
    recent_moods = [entry["mood"] for entry in mood_history[-5:] if isinstance(entry, dict) and "mood" in entry]
    
    stressed_count = sum(1 for mood in recent_moods if mood == "Sad")

    if stressed_count >= stress_threshold:
        print("⚠️ ALERT: Prolonged stress detected! Notifying HR.")

# 🎯 Function to recommend tasks based on detected mood
def recommend_task(mood):
    """
    Recommends an appropriate task based on the detected mood.
    Returns: A randomly chosen task suggestion (str)
    """
    mood = mood.capitalize()
    tasks = {
        "Happy": ["Collaborate on a new project", "Share positivity with the team"],
        "Sad": ["Take a break", "Listen to music", "Talk to a friend"],
        "Fear": ["Practice deep breathing", "Engage in light work", "Seek support"],
        "Angry": ["Cool down with a short walk", "Work on solo tasks"],
        "Surprise": ["Reflect on new insights", "Plan next steps"]
    }
    return random.choice(tasks.get(mood, ["No suggestion available"]))

# 🔒 Function to anonymize mood data for privacy
def anonymize_data(data):
    """
    Hashes mood data to protect employee privacy.
    Returns: An anonymized version of the mood string.
    """
    return hashlib.sha256(data.encode()).hexdigest()

# 💾 Function to save mood history to a JSON file
def save_mood_history(mood_history, filename="mood_history.json"):
    """
    Saves employee mood history, storing only employee_id and anonymized mood.
    This ensures privacy while keeping track of overall sentiment trends.
    """
    try:
        filtered_history = [
            {"employee_id": entry["employee_id"], "anonymized_mood": entry["anonymized_mood"]}
            for entry in mood_history
        ]

        with open(filename, "w") as file:
            json.dump(filtered_history, file, indent=4)  

        print("✅ Mood history successfully updated with anonymized data!")
    except Exception as e:
        print(f"❌ Error saving mood history: {e}")

# 📂 Function to load mood history from JSON file
def load_mood_history(filename="mood_history.json"):
    """
    Loads the saved mood history from JSON file.
    If the file is missing or corrupted, returns an empty list.
    """
    if not os.path.exists(filename) or os.stat(filename).st_size == 0:
        return []  
    
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        print("⚠️ Warning: mood_history.json is corrupted or empty. Resetting...")
        return []

# 📊 Function to visualize team mood distribution
def plot_team_mood(mood_history):
    """
    Generates a bar chart showing the distribution of moods among employees.
    Helps visualize team sentiment trends over time.
    """
    if not mood_history:
        print("⚠️ No mood history available to plot.")
        return
    
    mood_counts = Counter(entry["mood"] for entry in mood_history if isinstance(entry, dict) and "mood" in entry)
    
    if not mood_counts:
        print("⚠️ No valid mood data available for plotting.")
        return

    # Define colors for each mood type
    colors = {
        "Happy": "green",
        "Sad": "black",
        "Fear": "red",
        "Surprise": "orange",
        "Angry": "blue",
        "Neutral": "gold"
    }

    moods, counts = zip(*mood_counts.items())
    bar_colors = [colors.get(mood, "gray") for mood in moods]

    plt.bar(moods, counts, color=bar_colors)
    plt.xlabel("Mood")
    plt.ylabel("Count")
    plt.title("Team Mood Analysis")
    plt.show()

# 🏁 Main function to execute the program
def main():
    """
    Main program loop that:
      1. Captures video feed from webcam
      2. Detects facial emotion in real-time
      3. Logs anonymized mood history
      4. Provides task recommendations based on mood
      5. Monitors for prolonged stress
      6. Displays a mood analysis chart
    """
    cap = cv2.VideoCapture(0)  # Start webcam
    mood_history = load_mood_history()  # Load previous mood data
    
    employee_id = input("Enter Employee ID: ").strip()  # Get Employee ID

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detected_mood = detect_facial_emotion(frame)  # Detect mood
        
        # Store mood with anonymization
        mood_entry = {
            "employee_id": employee_id,
            "mood": detected_mood,  
            "anonymized_mood": anonymize_data(detected_mood)
        }
        
        mood_history.append(mood_entry)  
        save_mood_history(mood_history)  
        monitor_stress(mood_history)  # Check for stress alerts
        
        print(f"🧐 Employee: {employee_id} | Detected Mood: {detected_mood}")
        print(f"🎯 Recommended Task: {recommend_task(detected_mood)}")
        
        cv2.imshow('Real-Time Emotion Detection', frame)  # Display webcam feed
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    plot_team_mood(mood_history)  # Show mood distribution chart

# 🚀 Run the program if executed directly
if __name__ == "__main__":
    main()

