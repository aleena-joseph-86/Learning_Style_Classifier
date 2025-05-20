import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

class LearningStyleClassifier:
    def __init__(self):
        self.model = DecisionTreeClassifier(random_state=42)
        self.scaler = StandardScaler()
        self.learning_styles = ['Visual', 'Auditory', 'Kinesthetic']
        
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic data for training the model"""
        np.random.seed(42)
        
        # Features based on learning preferences
        data = {
            'prefers_visual_aids': np.random.randint(1, 6, n_samples),  # 1-5 scale
            'enjoys_discussions': np.random.randint(1, 6, n_samples),
            'likes_hands_on': np.random.randint(1, 6, n_samples),
            'remembers_images': np.random.randint(1, 6, n_samples),
            'prefers_lectures': np.random.randint(1, 6, n_samples),
            'enjoys_group_work': np.random.randint(1, 6, n_samples),
            'likes_demonstrations': np.random.randint(1, 6, n_samples),
            'prefers_written_instructions': np.random.randint(1, 6, n_samples),
            'enjoys_presentations': np.random.randint(1, 6, n_samples),
            'likes_experiments': np.random.randint(1, 6, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Generate labels based on feature combinations
        visual_score = (df['prefers_visual_aids'] + df['remembers_images'] + 
                       df['prefers_written_instructions']) / 3
        auditory_score = (df['enjoys_discussions'] + df['prefers_lectures'] + 
                         df['enjoys_presentations']) / 3
        kinesthetic_score = (df['likes_hands_on'] + df['enjoys_group_work'] + 
                           df['likes_demonstrations'] + df['likes_experiments']) / 4
        
        # Assign learning style based on highest score
        scores = np.column_stack((visual_score, auditory_score, kinesthetic_score))
        labels = np.argmax(scores, axis=1)
        
        return df, labels
    
    def train_model(self):
        """Train the model using synthetic data"""
        X, y = self.generate_synthetic_data()
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, y_train)
        return X_test, y_test
    
    def predict_learning_style(self, features):
        """Predict learning style based on input features"""
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return self.learning_styles[prediction[0]]
    
    def get_learning_recommendations(self, learning_style):
        """Get personalized learning recommendations based on learning style"""
        recommendations = {
            'Visual': [
                'Use mind maps and diagrams',
                'Watch educational videos',
                'Create visual notes with colors and symbols',
                'Use flashcards with images',
                'Study in a visually organized environment'
            ],
            'Auditory': [
                'Record and listen to lectures',
                'Participate in group discussions',
                'Read material aloud',
                'Use verbal mnemonics',
                'Listen to educational podcasts'
            ],
            'Kinesthetic': [
                'Take frequent study breaks',
                'Use hands-on activities',
                'Practice with physical models',
                'Study while walking or moving',
                'Create physical flashcards'
            ]
        }
        return recommendations[learning_style]

def main():
    st.title("Learning Style Classifier 🧠")
    st.write("Discover your learning style and get personalized recommendations!")
    
    # Initialize classifier
    classifier = LearningStyleClassifier()
    classifier.train_model()
    
    # Create input form
    st.subheader("Answer these questions (1-5 scale):")
    
    col1, col2 = st.columns(2)
    
    with col1:
        prefers_visual_aids = st.slider("I prefer learning with visual aids", 1, 5, 3)
        enjoys_discussions = st.slider("I enjoy group discussions", 1, 5, 3)
        likes_hands_on = st.slider("I prefer hands-on activities", 1, 5, 3)
        remembers_images = st.slider("I remember information better when presented visually", 1, 5, 3)
        prefers_lectures = st.slider("I learn well from lectures", 1, 5, 3)
    
    with col2:
        enjoys_group_work = st.slider("I enjoy group work", 1, 5, 3)
        likes_demonstrations = st.slider("I learn well from demonstrations", 1, 5, 3)
        prefers_written = st.slider("I prefer written instructions", 1, 5, 3)
        enjoys_presentations = st.slider("I enjoy giving presentations", 1, 5, 3)
        likes_experiments = st.slider("I enjoy experiments and practical work", 1, 5, 3)
    
    if st.button("Analyze Learning Style"):
        # Prepare features
        features = np.array([[
            prefers_visual_aids, enjoys_discussions, likes_hands_on,
            remembers_images, prefers_lectures, enjoys_group_work,
            likes_demonstrations, prefers_written, enjoys_presentations,
            likes_experiments
        ]])
        
        # Get prediction
        learning_style = classifier.predict_learning_style(features)
        
        # Display results
        st.subheader("Your Learning Style")
        st.write(f"Based on your responses, you are a **{learning_style} Learner**")
        
        # Display recommendations
        st.subheader("Personalized Learning Recommendations")
        recommendations = classifier.get_learning_recommendations(learning_style)
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Create visualization
        st.subheader("Learning Style Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        data = pd.DataFrame({
            'Visual': [prefers_visual_aids, remembers_images, prefers_written],
            'Auditory': [enjoys_discussions, prefers_lectures, enjoys_presentations],
            'Kinesthetic': [likes_hands_on, enjoys_group_work, likes_demonstrations]
        })
        
        sns.boxplot(data=data)
        plt.title("Your Learning Style Preferences")
        plt.ylabel("Preference Score")
        st.pyplot(fig)

if __name__ == "__main__":
    main() 