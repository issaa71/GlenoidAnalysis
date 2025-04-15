import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Glenoid Morphology Classifier",
    page_icon="🩻",
    layout="wide"
)

# Load the dataset
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('Feature Table.csv')
        return df
    except FileNotFoundError:
        st.error("Error: 'Feature Table.csv' not found. Please ensure the file is in the repository.")
        st.stop()

# Create the classification system
class GlenoidClassifier:
    def __init__(self):
        # Load and prepare data
        self.df = load_data()
        
        # Create features including the engineered ones
        self._prepare_features()
        
        # Train all the models with optimal hyperparameters
        self._train_models()
    
    def _prepare_features(self):
        # Create Main_Type column if not exists
        if 'Main_Type' not in self.df.columns:
            self.df['Main_Type'] = self.df['Label 1'].apply(
                lambda x: x[0] if x != 'Normal' else x
            )
        
        # Create engineered features based on the study
        self.df['Subluxation_Index'] = np.sqrt(
            np.power(self.df['AP Subluxation'], 2) + 
            np.power(self.df['SI Subluxation'], 2)
        )
        
        self.df['Version_Inclination_Vector'] = np.sqrt(
            np.power(self.df['Version'], 2) + 
            np.power(self.df['Inclination'], 2)
        )
        
        self.df['AP_SI_Ratio'] = self.df['AP Subluxation'] / self.df['SI Subluxation']
        
        self.df['Area_Radius_Ratio'] = self.df['Glenoid Surface Area'] / (
            np.pi * np.power(self.df['Sphere Radius'], 2)
        )
        
        self.df['Depth_Width_Ratio'] = self.df['Depth'] / self.df['Width']
        
        # Define the key features based on importance analysis
        self.features = [
            'Version', 'SI Subluxation', 'AP Subluxation', 'Inclination',
            'Subluxation_Index', 'Version_Inclination_Vector', 
            'AP_SI_Ratio', 'Area_Radius_Ratio', 'Depth_Width_Ratio'
        ]
        
        # Create scalers for each tier
        self.scalers = {}
        
        # Create pathologic subset
        pathologic_mask = self.df['Main_Type'] != 'Normal'
        self.pathologic_df = self.df[pathologic_mask]
        
        # Create type B and E subsets
        self.B_df = self.df[self.df['Main_Type'] == 'B']
        self.E_df = self.df[self.df['Main_Type'] == 'E']
    
    def _train_models(self):
        # Initialize the models dictionary
        self.models = {}
        
        # Tier 1: Normal vs. Pathologic (XGBoost)
        X = self.df[self.features].values
        y_normal = (self.df['Main_Type'] == 'Normal').astype(int)
        
        self.scalers['tier1'] = StandardScaler()
        X_scaled = self.scalers['tier1'].fit_transform(X)
        
        # Using the exact hyperparameters from the notebook
        self.models['tier1'] = XGBClassifier(
            learning_rate=0.05,
            max_depth=4,
            n_estimators=200,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        )
        self.models['tier1'].fit(X_scaled, y_normal)
        
        # Tier 2: Main Type Classification
        if len(self.pathologic_df) > 0:
            X_pathologic = self.pathologic_df[self.features].values
            self.scalers['tier2'] = StandardScaler()
            X_pathologic_scaled = self.scalers['tier2'].fit_transform(X_pathologic)
            
            # Type A vs. not A (Random Forest)
            y_A = (self.pathologic_df['Main_Type'] == 'A').astype(int)
            self.models['A_vs_not_A'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=42
            )
            self.models['A_vs_not_A'].fit(X_pathologic_scaled, y_A)
            
            # Type B vs. not B (Random Forest)
            y_B = (self.pathologic_df['Main_Type'] == 'B').astype(int)
            self.models['B_vs_not_B'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=42
            )
            self.models['B_vs_not_B'].fit(X_pathologic_scaled, y_B)
            
            # Type E vs. not E (SVM)
            y_E = (self.pathologic_df['Main_Type'] == 'E').astype(int)
            self.models['E_vs_not_E'] = SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
            self.models['E_vs_not_E'].fit(X_pathologic_scaled, y_E)
        
        # Tier 3: Subtype Classification
        
        # Type B: B2 vs. B3
        if len(self.B_df) > 0:
            X_B = self.B_df[self.features].values
            self.scalers['tier3_B'] = StandardScaler()
            X_B_scaled = self.scalers['tier3_B'].fit_transform(X_B)
            
            y_B2_vs_B3 = (self.B_df['Label 1'] == 'B2').astype(int)
            self.models['B2_vs_B3'] = SVC(
                C=1.0,
                kernel='rbf',
                probability=True,
                random_state=42
            )
            self.models['B2_vs_B3'].fit(X_B_scaled, y_B2_vs_B3)
        
        # Type E: E2 vs. E3
        if len(self.E_df) > 0:
            X_E = self.E_df[self.features].values
            self.scalers['tier3_E'] = StandardScaler()
            X_E_scaled = self.scalers['tier3_E'].fit_transform(X_E)
            
            y_E2_vs_E3 = (self.E_df['Label 1'] == 'E2').astype(int)
            self.models['E2_vs_E3'] = XGBClassifier(
                learning_rate=0.1,
                max_depth=5,
                n_estimators=50,
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42
            )
            self.models['E2_vs_E3'].fit(X_E_scaled, y_E2_vs_E3)
    
    def classify(self, measurements):
        """
        Classify a new glenoid based on given measurements
        
        Parameters:
        measurements (dict): Dictionary with the following keys:
            'Version' - Version angle in degrees
            'SI_Subluxation' - SI subluxation in mm
            'AP_Subluxation' - AP subluxation in mm
            'Inclination' - Inclination in degrees
            
        Returns:
        dict: Classification results
        """
        # Prepare input features
        input_data = self._prepare_input(measurements)
        
        # Tier 1: Normal vs. Pathologic
        tier1_result, tier1_confidence = self._classify_tier1(input_data)
        
        results = {
            'tier1': {
                'result': tier1_result,
                'confidence': tier1_confidence
            },
            'tier2': {
                'result': None,
                'confidence': None,
                'probabilities': {}
            },
            'tier3': {
                'result': None,
                'confidence': None
            },
            'final': {
                'type': None,
                'confidence': None
            }
        }
        
        # Only proceed to Tier 2 if pathologic
        if tier1_result == "Pathologic":
            # Tier 2: Main Type Classification
            tier2_result, tier2_confidence, type_probs = self._classify_tier2(input_data)
            results['tier2']['result'] = tier2_result
            results['tier2']['confidence'] = tier2_confidence
            results['tier2']['probabilities'] = type_probs
            
            # Tier 3: Subtype Classification
            if tier2_result in ['A', 'B', 'E']:
                tier3_result, tier3_confidence = self._classify_tier3(input_data, tier2_result)
                results['tier3']['result'] = tier3_result
                results['tier3']['confidence'] = tier3_confidence
                
                # Set final result
                results['final']['type'] = tier3_result
                results['final']['confidence'] = tier1_confidence * tier2_confidence * tier3_confidence
            else:
                results['final']['type'] = "Unknown"
                results['final']['confidence'] = 0.0
        else:
            # Normal glenoid
            results['final']['type'] = "Normal"
            results['final']['confidence'] = tier1_confidence
        
        return results
    
    def _prepare_input(self, measurements):
        """Prepare input features from measurements"""
        input_features = {}
        
        # Convert input keys to match dataframe column names
        key_mapping = {
            'Version': 'Version',
            'SI_Subluxation': 'SI Subluxation',
            'AP_Subluxation': 'AP Subluxation',
            'Inclination': 'Inclination'
        }
        
        for input_key, df_key in key_mapping.items():
            if input_key in measurements:
                input_features[df_key] = measurements[input_key]
        
        # Calculate engineered features
        input_features['Subluxation_Index'] = np.sqrt(
            np.power(input_features['AP Subluxation'], 2) + 
            np.power(input_features['SI Subluxation'], 2)
        )
        
        input_features['Version_Inclination_Vector'] = np.sqrt(
            np.power(input_features['Version'], 2) + 
            np.power(input_features['Inclination'], 2)
        )
        
        input_features['AP_SI_Ratio'] = input_features['AP Subluxation'] / input_features['SI Subluxation']
        
        # Some features might be missing for a real-world case
        # We'll use approximations for them based on study averages
        if 'Area_Radius_Ratio' not in input_features:
            input_features['Area_Radius_Ratio'] = 0.3  # Approximate average
        
        if 'Depth_Width_Ratio' not in input_features:
            input_features['Depth_Width_Ratio'] = 0.25  # Approximate average
        
        # Create a feature vector in the correct order
        X = np.array([[input_features.get(feature, 0) for feature in self.features]])
        
        return X
    
    def _classify_tier1(self, X):
        """Tier 1: Normal vs. Pathologic"""
        X_scaled = self.scalers['tier1'].transform(X)
        normal_prob = self.models['tier1'].predict_proba(X_scaled)[0, 1]
        
        if normal_prob > 0.5:
            return "Normal", normal_prob
        else:
            return "Pathologic", 1 - normal_prob
    
    def _classify_tier2(self, X):
        """Tier 2: Main Type Classification"""
        X_scaled = self.scalers['tier2'].transform(X)
        
        # Get probabilities for each type
        typeA_prob = self.models['A_vs_not_A'].predict_proba(X_scaled)[0, 1]
        typeB_prob = self.models['B_vs_not_B'].predict_proba(X_scaled)[0, 1]
        typeE_prob = self.models['E_vs_not_E'].predict_proba(X_scaled)[0, 1]
        
        # Normalize probabilities
        total_prob = typeA_prob + typeB_prob + typeE_prob
        typeA_prob /= total_prob
        typeB_prob /= total_prob
        typeE_prob /= total_prob
        
        # Determine the most likely type
        type_probs = {
            'A': typeA_prob,
            'B': typeB_prob,
            'E': typeE_prob
        }
        main_type = max(type_probs, key=type_probs.get)
        
        return main_type, type_probs[main_type], type_probs
    
    def _classify_tier3(self, X, main_type):
        """Tier 3: Subtype Classification"""
        if main_type == 'A':
            # Only subtype A2 in this dataset
            return "A2", 1.0
        
        elif main_type == 'B':
            X_scaled = self.scalers['tier3_B'].transform(X)
            b2_prob = self.models['B2_vs_B3'].predict_proba(X_scaled)[0, 1]
            
            if b2_prob > 0.5:
                return "B2", b2_prob
            else:
                return "B3", 1 - b2_prob
        
        elif main_type == 'E':
            X_scaled = self.scalers['tier3_E'].transform(X)
            e2_prob = self.models['E2_vs_E3'].predict_proba(X_scaled)[0, 1]
            
            if e2_prob > 0.5:
                return "E2", e2_prob
            else:
                return "E3", 1 - e2_prob
        
        return "Unknown", 0.0

def provide_clinical_interpretation(glenoid_type):
    """
    Provide clinical interpretation 
    """
    if glenoid_type == "Normal":
        return "• No significant glenoid erosion or deformity"
    
    elif glenoid_type == "A2":
        return "• Moderate to severe central erosion (4–8 mm of bone loss), centrally located on the glenoid"
    
    elif glenoid_type == "B2":
        return "• Biconcave posterior erosion: the normal concavity of the glenoid is split into two regions"
    
    elif glenoid_type == "B3":
        return "• Monoconcave posterior erosion: the entire glenoid surface is eroded posteriorly"
    
    elif glenoid_type == "E2":
        return "• Biconcave superior erosion: similar to B2 but located superiorly"

    elif glenoid_type == "E3":
        return "• Monoconcave superior erosion: similar to B3 but directed superiorly"
    
    return "• Interpretation not available for this type"

def plot_probability_chart(type_probs):
    """Create a bar chart of type probabilities"""
    fig, ax = plt.subplots(figsize=(6, 3))
    types = list(type_probs.keys())
    probs = list(type_probs.values())
    
    ax.bar(types, probs, color=['#3498db', '#2ecc71', '#e74c3c'])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Type Probability Distribution')
    
    # Add probability values on top of bars
    for i, prob in enumerate(probs):
        ax.text(i, prob + 0.02, f"{prob:.2f}", ha='center')
    
    return fig

# Initialize the app
st.title('🩻 Glenoid Morphology Classifier')
st.markdown("""
This application helps classify glenoid morphology based on key measurements. 
Enter the required values below to get a classification according to the Walch classification system.
""")

# Initialize the classifier on app startup
@st.cache_resource
def initialize_classifier():
    return GlenoidClassifier()

# Load classifier
with st.spinner('Initializing classifier...'):
    classifier = initialize_classifier()

# Create a form for input
st.subheader('📏 Input Measurements')
st.markdown('Enter the following measurements obtained from shoulder imaging:')

with st.form("measurement_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        version = st.number_input('Version angle (degrees, typical range -30 to 10)', 
                                 min_value=-30.0, max_value=30.0, value=0.0, 
                                 help='Glenoid version angle in degrees')
        
        si_subluxation = st.number_input('SI Subluxation (mm, typical range 10 to 80)',
                                       min_value=-10.0, max_value=10.0, value=0.0,
                                       help='Superior-Inferior subluxation in mm')
    
    with col2:
        ap_subluxation = st.number_input('AP Subluxation (mm, typical range 30 to 90)',
                                       min_value=-10.0, max_value=10.0, value=0.0,
                                       help='Anterior-Posterior subluxation in mm')
        
        inclination = st.number_input('Inclination (degrees, typical range -10 to 20)',
                                    min_value=-30.0, max_value=30.0, value=0.0,
                                    help='Glenoid inclination in degrees')
    
    submitted = st.form_submit_button("Classify Glenoid")

# Process the input and display results
if submitted:
    # Prepare the measurements
    measurements = {
        'Version': version,
        'SI_Subluxation': si_subluxation,
        'AP_Subluxation': ap_subluxation,
        'Inclination': inclination
    }
    
    # Classify the glenoid
    with st.spinner('Classifying...'):
        results = classifier.classify(measurements)
    
    # Display results
    st.subheader('🔍 Classification Results')
    
    # Create columns for organized display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Final result with highest emphasis
        final_type = results['final']['type']
        confidence = results['final']['confidence']
        
        # Display final classification in a colored box
        st.markdown(f"""
        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border: 1px solid #add8e6;">
            <h3 style="margin: 0;">Final Classification: {final_type}</h3>
            <p style="margin: 5px 0 0 0;">Confidence: {confidence:.1%}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Clinical interpretation
        st.subheader('📋 Clinical Considerations')
        interpretation = provide_clinical_interpretation(final_type)
        st.markdown(f"<div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px;'>{interpretation}</div>", 
                    unsafe_allow_html=True)
        
    with col2:
        # Show probability chart for pathologic types
        if results['tier2']['probabilities']:
            st.markdown("### Type Probabilities")
            fig = plot_probability_chart(results['tier2']['probabilities'])
            st.pyplot(fig)
    
    # Detailed results in an expandable section
    with st.expander("View Detailed Classification Process"):
        st.markdown("#### Tier 1: Normal vs. Pathologic")
        st.markdown(f"Result: **{results['tier1']['result']}**")
        st.markdown(f"Confidence: {results['tier1']['confidence']:.1%}")
        
        if results['tier2']['result']:
            st.markdown("#### Tier 2: Main Type Classification")
            st.markdown(f"Result: **Type {results['tier2']['result']}**")
            st.markdown(f"Confidence: {results['tier2']['confidence']:.1%}")
            
            # Show all type probabilities
            st.markdown("Type probabilities:")
            for type_letter, prob in results['tier2']['probabilities'].items():
                st.markdown(f"- Type {type_letter}: {prob:.1%}")
        
        if results['tier3']['result']:
            st.markdown("#### Tier 3: Specific Subtype")
            st.markdown(f"Result: **{results['tier3']['result']}**")
            st.markdown(f"Confidence: {results['tier3']['confidence']:.1%}")

# Add information about the classification system
with st.expander("About the Walch Glenoid Classification System"):
    st.markdown("""
    The Walch Classification is a system used to categorize glenoid morphology in shoulders with glenohumeral osteoarthritis:

    - **Type A**: Centered humeral head with minor erosion
      - A1: Minor central erosion
      - A2: Major central erosion

    - **Type B**: Posterior subluxation of the humeral head
      - B1: Posterior joint narrowing, no erosion
      - B2: Biconcave posterior erosion: the normal concavity of the glenoid is split into two regions
      - B3: Monoconcave posterior erosion: the entire glenoid surface is eroded posteriorly

    - **Type C**: Dysplastic glenoid with severe retroversion (>25°)

    - **Type D**: Glenoid with significant anterior erosion (uncommon)

    - **Type E**: Global concentric wear 
      - E2: Biconcave superior erosion: similar to B2 but located superiorly
      - E3: Monoconcave superior erosion: similar to B3 but directed superiorly
    """)

# Add footer
st.markdown("---")
st.markdown("© 2025 Glenoid Morphology Classifier | Created with Streamlit")
