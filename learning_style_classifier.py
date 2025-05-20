import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Learning Style Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 5px;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

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
    st.title("🎓 Learning Style Classifier")
    st.markdown("""
    ### Discover Your Learning Style
    This tool helps you understand your preferred learning style and provides personalized recommendations 
    to enhance your learning experience. Answer the questions below to get started!
    """)
    
    # Initialize classifier
    classifier = LearningStyleClassifier()
    classifier.train_model()
    
    # Create input form with better organization
    st.markdown("### 📝 Learning Preferences Assessment")
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;'>
        <p style='margin-bottom: 10px;'><strong>How to answer:</strong></p>
        <p style='margin: 5px 0;'>1️⃣ = Strongly Disagree</p>
        <p style='margin: 5px 0;'>2️⃣ = Disagree</p>
        <p style='margin: 5px 0;'>3️⃣ = Neutral</p>
        <p style='margin: 5px 0;'>4️⃣ = Agree</p>
        <p style='margin: 5px 0;'>5️⃣ = Strongly Agree</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for better organization
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Learning Environment")
        prefers_visual_aids = st.slider(
            "I prefer learning with visual aids like diagrams and charts",
            1, 5, 3,
            help="How much do you enjoy learning through images, diagrams, and visual materials?"
        )
        enjoys_discussions = st.slider(
            "I learn best through group discussions and conversations",
            1, 5, 3,
            help="How much do you enjoy learning through conversations and discussions?"
        )
        likes_hands_on = st.slider(
            "I prefer learning through hands-on activities and experiments",
            1, 5, 3,
            help="How much do you enjoy learning through physical activities?"
        )
        remembers_images = st.slider(
            "I remember information better when it's presented visually",
            1, 5, 3,
            help="How well do you retain information that's presented visually?"
        )
        prefers_lectures = st.slider(
            "I learn effectively from lectures and verbal explanations",
            1, 5, 3,
            help="How effective are lectures for your learning?"
        )
    
    with col2:
        st.markdown("#### Learning Activities")
        enjoys_group_work = st.slider(
            "I enjoy collaborative learning and group projects",
            1, 5, 3,
            help="How much do you enjoy collaborative learning activities?"
        )
        likes_demonstrations = st.slider(
            "I learn well from practical demonstrations",
            1, 5, 3,
            help="How effective are practical demonstrations for your learning?"
        )
        prefers_written = st.slider(
            "I prefer written instructions and materials",
            1, 5, 3,
            help="How much do you prefer written materials over verbal instructions?"
        )
        enjoys_presentations = st.slider(
            "I enjoy giving presentations and explaining concepts verbally",
            1, 5, 3,
            help="How comfortable are you with presenting information verbally?"
        )
        likes_experiments = st.slider(
            "I enjoy learning through experiments and practical work",
            1, 5, 3,
            help="How much do you enjoy learning through experiments?"
        )
    
    st.markdown("---")
    
    if st.button("🔍 Analyze My Learning Style", use_container_width=True):
        with st.spinner("Analyzing your learning preferences..."):
            # Prepare features
            features = np.array([[
                prefers_visual_aids, enjoys_discussions, likes_hands_on,
                remembers_images, prefers_lectures, enjoys_group_work,
                likes_demonstrations, prefers_written, enjoys_presentations,
                likes_experiments
            ]])
            
            # Get prediction
            learning_style = classifier.predict_learning_style(features)
            
            # Display results in a nice container
            st.markdown("## 📊 Results")
            
            # Create a container for the learning style result
            with st.container():
                st.markdown(f"""
                <div style='background-color: #e8f5e9; padding: 20px; border-radius: 10px; margin: 10px 0;'>
                    <h3 style='color: #2e7d32; margin: 0;'>Your Learning Style: {learning_style} Learner</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a progress bar for each learning style
                visual_score = (prefers_visual_aids + remembers_images + prefers_written) / 3
                auditory_score = (enjoys_discussions + prefers_lectures + enjoys_presentations) / 3
                kinesthetic_score = (likes_hands_on + enjoys_group_work + likes_demonstrations + likes_experiments) / 4
                
                st.markdown("#### Learning Style Distribution")
                st.progress(visual_score / 5, text="Visual")
                st.progress(auditory_score / 5, text="Auditory")
                st.progress(kinesthetic_score / 5, text="Kinesthetic")
            
            # Display recommendations in a nice format
            st.markdown("## 💡 Personalized Learning Recommendations")
            recommendations = classifier.get_learning_recommendations(learning_style)
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"""
                <div style='padding: 15px; margin: 10px 0; background-color: #f0f2f6; border-radius: 10px; border-left: 5px solid #4CAF50;'>
                    <strong style='color: #2e7d32;'>{i}.</strong> {rec}
                </div>
                """, unsafe_allow_html=True)
            
            # Create visualization with better styling
            st.markdown("## 📈 Learning Style Analysis")
            fig, ax = plt.subplots(figsize=(10, 6))
            data = pd.DataFrame({
                'Visual': [prefers_visual_aids, remembers_images, prefers_written],
                'Auditory': [enjoys_discussions, prefers_lectures, enjoys_presentations],
                'Kinesthetic': [likes_hands_on, enjoys_group_work, likes_demonstrations]
            })
            
            sns.set_style("whitegrid")
            sns.boxplot(data=data, palette="Set2")
            plt.title("Your Learning Style Preferences", pad=20)
            plt.ylabel("Preference Score")
            plt.xticks(rotation=45)
            st.pyplot(fig)

if __name__ == "__main__":
    main() 