import streamlit as st
import pandas as pd
import pickle
import warnings
import sklearn
warnings.filterwarnings('ignore')

st.title('Fetal Health Classification: A Machine Learning App 👶')
st.image('fetal_health_image.gif', use_column_width=True, caption="Utilize advanced machine learning application to predict health classification!")
st.write("This app uses multiple inputs to predict the health of fetuses") 
st.divider()

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file) if uploaded_file is not None else None

#Chat GPT helped with this
def highlight_predicted_class(row):
    color = ''
    if row['Predicted Class'] == 'Normal':
        color = 'lime'
    elif row['Predicted Class'] == 'Suspect':
        color = 'yellow'
    elif row['Predicted Class'] == 'Pathological':
        color = 'orange'
    return [f'background-color: {color}' if col == 'Predicted Class' else '' for col in row.index]

model_selection = None
default_df = pd.read_csv('fetal_health.csv')
default_df.head()

with st.sidebar.form("user_input_form"):
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], help="Upload your CSV file with fetal health details.")
    if uploaded_file:
        st.dataframe(pd.read_csv(uploaded_file))
    model_selection = st.checkbox("Select which model you'd like to utilize:", ['Decision Tree','Random Forest','Ada Boost','Soft Voting (Ensemble)'])
    submit_button = st.form_submit_button("Predict")


if model_selection:
    if model_selection == 'Decision Tree':
        dt_pickle = open('DT_ML.pickle', 'rb') 
        clf = pickle.load(dt_pickle) 
        dt_pickle.close()
    elif model_selection == 'Random Forest':
        rf_pickle = open('RF_ML.pickle', 'rb') 
        clf = pickle.load(rf_pickle) 
        rf_pickle.close()
    elif model_selection == 'Ada Boost':
        ada_pickle = open('ADA_ML.pickle', 'rb') 
        clf = pickle.load(ada_pickle) 
        ada_pickle.close()
    elif model_selection == 'Soft Voting (Ensemble)':
        sv_pickle = open('SV_ML.pickle', 'rb') 
        clf = pickle.load(sv_pickle) 
        sv_pickle.close()
    else:
        st.write("Please Select a Model")



if uploaded_file:
    # Chat GPT helped with this styling
    st.markdown("<h4 style='color: green;'>✅ CSV Successfully uploaded.</h4>", unsafe_allow_html=True)
    original_df = load_data(uploaded_file)
    encode_df = default_df.copy().drop(columns=['fetal_health'], errors='ignore')
    encode_df = pd.concat([encode_df, original_df])
    encode_dummy_df = pd.get_dummies(encode_df)
    appended_rows = encode_dummy_df.tail(len(original_df))

    predictions = []
    normal_probs = []
    suspect_probs = []
    pathological_probs = []

    for i in range(len(appended_rows)):
        row = appended_rows.iloc[[i]]

        predicted_class = clf.predict(row)[0]

        class_probs = clf.predict_proba(row)[0]

        predictions.append(predicted_class)
        normal_probs.append(f"{class_probs[0] * 100:.2f}%")
        suspect_probs.append(f"{class_probs[1] * 100:.2f}%")
        pathological_probs.append(f"{class_probs[2] * 100:.2f}%")

    original_df["Predicted Class"] = predictions
    original_df["Normal Probability"] = normal_probs
    original_df["Suspect Probability"] = suspect_probs
    original_df["Pathological Probability"] = pathological_probs

    st.write("### Predicted Classes and Probabilities")

    #Chat GPT helped with implementation of it as well
    st.dataframe(original_df.style.apply(highlight_predicted_class, axis=1))
else:
    # Chat GPT helped with this styling
    st.markdown("<h4 style='color: red;'>❌ Please upload a CSV file to predict.</h4>", unsafe_allow_html=True)


if uploaded_file: 
    if model_selection == 'Decision Tree':

        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3, tab4 = st.tabs(["Decision Tree", "Feature Importance", "Confusion Matrix", "Classification Report"])

        # Tab 1: Visualizing Decision Tree
        with tab1:
            st.write("### Decision Tree Visualization")
            st.image('dt_vis.svg')
            st.caption("Visualization of the Decision Tree used in prediction.")

        # Tab 2: Feature Importance Visualization
        with tab2:
            st.write("### Feature Importance")
            st.image('DT_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 3: Confusion Matrix
        with tab3:
            st.write("### Confusion Matrix")
            st.image('cmdt.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab4:
            st.write("### Classification Report")
            report_df = pd.read_csv('DT.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health level.")

    if model_selection == 'Random Forest':
        
        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

    # Tab 2: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('RF_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 3: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('cmrf.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('RF.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health level.")

    if model_selection == 'Ada Boost':
        
        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

    # Tab 2: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('Ada_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 3: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('cmada.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('ada.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health level.")

    if model_selection == 'Soft Voting (Ensemble)':
        
        # Showing additional items in tabs
        st.subheader("Prediction Performance")
        tab1, tab2, tab3 = st.tabs(["Feature Importance", "Confusion Matrix", "Classification Report"])

    # Tab 2: Feature Importance Visualization
        with tab1:
            st.write("### Feature Importance")
            st.image('SV_imp.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

        # Tab 3: Confusion Matrix
        with tab2:
            st.write("### Confusion Matrix")
            st.image('cmsv.svg')
            st.caption("Confusion Matrix of model predictions.")

        # Tab 4: Classification Report
        with tab3:
            st.write("### Classification Report")
            report_df = pd.read_csv('SV.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each health level.")

