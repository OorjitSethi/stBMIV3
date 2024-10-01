import streamlit as st
import pandas as pd
import numpy as np
import cv2
import dlib
import joblib
import os
from imutils import face_utils
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def load_and_train_model():
    # Check if the model already exists
    model_path = 'trained_model.joblib'
    scaler_path = 'scaler.joblib'
    bmi_scaler_path = 'bmi_scaler.joblib'
    feature_cols_path = 'feature_cols.joblib'
    high_corr_features_path = 'high_corr_features.joblib'
    data_path = 'data.joblib'
    
    if os.path.exists(model_path):
        # Load the model and other objects
        best_rf = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        bmi_scaler = joblib.load(bmi_scaler_path)
        feature_cols = joblib.load(feature_cols_path)
        high_corr_features = joblib.load(high_corr_features_path)
        data = joblib.load(data_path)
        st.success("Model loaded successfully.")
        return best_rf, scaler, bmi_scaler, feature_cols, high_corr_features, data
    else:
        # Initialize progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
    
        # Step 1: Load the dataset
        status_text.text("Loading dataset...")
        data = pd.read_csv('BMI_FM_PCs.txt', sep='\t')
        progress_bar.progress(5)
        
        # Step 2: Drop rows with missing values
        status_text.text("Cleaning data...")
        data = data.dropna()
        progress_bar.progress(10)
        
        # Step 3: Define feature columns
        status_text.text("Preparing features...")
        excluded_features = ['ID', 'age', 'bmi', 'height', 'weight']
        feature_cols = data.columns.drop(excluded_features)
        progress_bar.progress(15)
        
        # Step 4: Initialize and fit the scaler for features
        status_text.text("Scaling features...")
        scaler = StandardScaler()
        scaler.fit(data[feature_cols])
        data_scaled = data.copy()
        data_scaled[feature_cols] = scaler.transform(data[feature_cols])
        progress_bar.progress(20)
        
        # Step 5: Scale the BMI
        status_text.text("Scaling BMI...")
        bmi_scaler = StandardScaler()
        data_scaled['bmi'] = bmi_scaler.fit_transform(data[['bmi']])
        progress_bar.progress(25)
        
        # Step 6: Compute correlation matrix
        status_text.text("Computing correlations...")
        corr_matrix = data_scaled.corr()
        corr_with_bmi = corr_matrix.loc[feature_cols, 'bmi']
        high_corr_features = corr_with_bmi[abs(corr_with_bmi) > 0.1].index.tolist()
        high_corr_features = [feat for feat in high_corr_features if feat not in ['age', 'weight']]
        progress_bar.progress(30)
        
        # Step 7: Define X and y
        status_text.text("Preparing training data...")
        X = data_scaled[high_corr_features]
        y = data_scaled['bmi']
        progress_bar.progress(35)
        
        # Step 8: Split the data
        status_text.text("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        progress_bar.progress(40)
        
        # Step 9: Initialize and train the model
        status_text.text("Training model...")
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        progress_bar.progress(50)
        
        # Step 10: Perform Grid Search
        status_text.text("Optimizing model with Grid Search...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        progress_bar.progress(100)
        
        # Step 11: Finalizing
        status_text.text("Finalizing model...")
        
        # Save the model and other objects
        joblib.dump(best_rf, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(bmi_scaler, bmi_scaler_path)
        joblib.dump(feature_cols, feature_cols_path)
        joblib.dump(high_corr_features, high_corr_features_path)
        joblib.dump(data, data_path)
        
        # Clear progress bar and status text
        progress_bar.empty()
        status_text.empty()
        
        st.success("Model trained and saved successfully.")
        
        return best_rf, scaler, bmi_scaler, feature_cols, high_corr_features, data

    # If model loading failed
    st.error("Failed to load or train the model.")
    return None, None, None, None, None, None

def get_facial_landmarks(image, predictor_path='shape_predictor_68_face_landmarks.dat'):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    
    if image is None:
        raise ValueError("Image not found or unable to read")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 1)
    
    if len(rects) == 0:
        raise ValueError("No face detected in the image")
    
    rect = rects[0]
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    
    return shape, image

def compute_facial_metrics(landmarks):
    left_cheek = landmarks[1]
    right_cheek = landmarks[15]
    cheek_width = np.linalg.norm(left_cheek - right_cheek)
    
    left_jaw = landmarks[3]
    right_jaw = landmarks[13]
    jaw_width = np.linalg.norm(left_jaw - right_jaw)
    
    cheek_to_jaw_ratio = cheek_width / jaw_width if jaw_width != 0 else 0
    
    chin = landmarks[8]
    forehead = landmarks[27]
    face_height = np.linalg.norm(chin - forehead)
    
    face_width = cheek_width
    width_to_height_ratio = face_width / face_height if face_height != 0 else 0
    
    face_area = np.pi * (face_width / 2) * (face_height / 2)
    parameter_area_ratio = face_area  # Adjust as needed
    
    return {
        'WtoHRatio': width_to_height_ratio,
        'CheektoJawWidth': cheek_to_jaw_ratio,
        'ParameterAreaRatio': parameter_area_ratio
    }

def prepare_feature_vector(facial_metrics, feature_cols, data):
    feature_vector = pd.DataFrame(columns=feature_cols)
    feature_means = data[feature_cols].mean()
    feature_vector.loc[0] = feature_means
    for feature in facial_metrics:
        if feature in feature_vector.columns:
            feature_vector.at[0, feature] = facial_metrics[feature]
    return feature_vector

def scale_features(feature_vector, scaler):
    scaled_features = scaler.transform(feature_vector)
    scaled_feature_vector = pd.DataFrame(scaled_features, columns=feature_vector.columns)
    return scaled_feature_vector

def predict_bmi(scaled_feature_vector, best_rf, bmi_scaler, high_corr_features):
    bmi_prediction_scaled = best_rf.predict(scaled_feature_vector[high_corr_features])
    bmi_original_scale = bmi_scaler.inverse_transform(bmi_prediction_scaled.reshape(-1, 1))
    return bmi_original_scale[0][0]

def estimate_bmi_from_image(image, best_rf, scaler, bmi_scaler, feature_cols, high_corr_features, data):
    landmarks, image = get_facial_landmarks(image)
    facial_metrics = compute_facial_metrics(landmarks)
    feature_vector = prepare_feature_vector(facial_metrics, feature_cols, data)
    scaled_feature_vector = scale_features(feature_vector, scaler)
    bmi_prediction = predict_bmi(scaled_feature_vector, best_rf, bmi_scaler, high_corr_features)
    return bmi_prediction

def main():
    st.title('BMI Prediction from Facial Features (Beta Version)')
    st.write("This is a beta version with a percentage uncertainty of around ±0.72%.")
    
    # Display sample image
    st.write("**Sample Image:**")
    sample_image_path = 'sample_image.jpg'  # Ensure this file is in your working directory
    sample_image = cv2.imread(sample_image_path)
    if sample_image is not None:
        st.image(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB), caption='Sample Image', width=300)
    else:
        st.write("Sample image not found.")
    
    # File uploader
    st.write("Please upload an image similar to the sample image above.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Load and display the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Uploaded Image', width=300)
        
        # Display a spinner while processing
        with st.spinner('Estimating BMI...'):
            # Load and train the model
            best_rf, scaler, bmi_scaler, feature_cols, high_corr_features, data = load_and_train_model()
            
            # Perform predictions and display the result
            try:
                bmi_prediction = estimate_bmi_from_image(
                    image, best_rf, scaler, bmi_scaler, feature_cols, high_corr_features, data
                )
                # Display BMI in larger font size
                bmi_display = f"<h2 style='text-align: center;'>Predicted BMI: {bmi_prediction:.2f}</h2>"
                st.markdown(bmi_display, unsafe_allow_html=True)
                
                # Button to copy BMI to clipboard
                copy_bmi = st.button("Copy BMI to Clipboard")
                if copy_bmi:
                    st.write("Press Ctrl+C (Cmd+C on Mac) to copy the BMI below:")
                    st.code(f"{bmi_prediction:.2f}", language='')

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
