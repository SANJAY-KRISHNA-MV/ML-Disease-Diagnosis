from flask import Flask, render_template, request
import numpy as np
import statistics
import joblib  # or import pickle if you used pickle to save your models

# --- Load your trained models ---
# Assuming your .pkl files are in the same directory as app.py
# or adjust the paths accordingly
MODEL_FOLDER = './models/' # Create a folder named 'models' and put your .pkl files there (optional but good practice)
final_rf_model = joblib.load(MODEL_FOLDER + 'final_rf_model.pkl')  # Replace 'final_rf_model.pkl' with your actual filename
final_nb_model = joblib.load(MODEL_FOLDER + 'final_nb_model.pkl')  # Replace 'final_nb_model.pkl' with your actual filename
final_svm_model = joblib.load(MODEL_FOLDER + 'final_svm_model.pkl')  # Replace 'final_svm_model.pkl' with your actual filename
# --- End of Model Loading ---

# Placeholder for X.columns.values -  REPLACE with your actual symptoms list if available
symptoms_list = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain',
'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination','fatigue',
'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat',
'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion',
'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_eyes', 'back_pain', 'constipation',
'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload',
'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps',
'bruising', 'obesity', 'swollen_legs', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
'weakness_of_one_body_side', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
'history_of_alcohol_consumption', 'fluid_in_lungs', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples',
'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
'yellow_crust_ooze']
symptoms = symptoms_list #X.columns.values # replace X.columns.values with symptoms_list for now


# Placeholder for encoder.classes_ -  REPLACE with your actual encoder classes if available
encoder_classes_list = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
 'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes ',
 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension ', 'Migraine',
 'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
 'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
 'Dimorphic hemmorhoids(piles)', 'Heart attack', 'Varicose veins', 'Hypothyroidism',
 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis', 'Arthritis',
 '(vertigo) Paroxysmal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis', 'Impetigo']
encoder_classes_ = encoder_classes_list # encoder.classes_ # replace encoder.classes_ with encoder_classes_list for now


# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder_classes_ # encoder.classes_ replace with encoder_classes_
}

# Defining the Function
# Input: string containing symptoms separated by commas
# Output: Generated predictions by models
def predictDisease(symptoms_input): # Renamed parameter to avoid shadowing global name
    symptoms_input_list = symptoms_input.split(",") # Split input string into a list
    symptoms_input_list = [symptom.strip() for symptom in symptoms_input_list] # strip whitespace from symptoms

    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms_input_list:
        if symptom in data_dict["symptom_index"]: # Check if symptom is in index
            index = data_dict["symptom_index"][symptom]
            input_data[index] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not found in symptom index.") # Handle unknown symptom

    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)

    # generating individual outputs
    rf_prediction_index = final_rf_model.predict(input_data)[0]
    nb_prediction_index = final_nb_model.predict(input_data)[0]
    svm_prediction_index = final_svm_model.predict(input_data)[0]

    rf_prediction = data_dict["predictions_classes"][rf_prediction_index]
    nb_prediction = data_dict["predictions_classes"][nb_prediction_index]
    svm_prediction = data_dict["predictions_classes"][svm_prediction_index]


    # making final prediction by taking mode of all predictions
    # Use statistics.mode instead of scipy.stats.mode

    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction":final_prediction
    }
    return predictions


app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        symptoms_input = request.form['symptoms']
        predictions = predictDisease(symptoms_input)
        return render_template('result.html', predictions=predictions, symptoms_input=symptoms_input)

if __name__ == '__main__':
    app.run(debug=True)