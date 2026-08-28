import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediCost - Healthcare Cost Prediction",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #f4f7fb;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #12355b;
    margin-bottom: 5px;
}

.subtitle {
    color: #667085;
    font-size: 16px;
}

.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 18px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.login-box {
    max-width: 500px;
    margin: 80px auto;
    background-color: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0px 8px 30px rgba(0,0,0,0.12);
}

.result-box {
    background-color: #eefaf2;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    border: 2px solid #9ad6ae;
}

.result-money {
    font-size: 45px;
    font-weight: 900;
    color: #157347;
}

.footer {
    text-align: center;
    color: #667085;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "doctor_name" not in st.session_state:
    st.session_state.doctor_name = ""

if "history" not in st.session_state:
    st.session_state.history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    try:
        data = pd.read_csv("insurance.csv")

    except FileNotFoundError:

        try:
            data = pd.read_csv("insurance.txt")

        except FileNotFoundError:
            st.error(
                "Dataset not found. Keep insurance.csv "
                "in the same folder as this Python file."
            )
            st.stop()

    data.columns = [
        str(column).strip().lower()
        for column in data.columns
    ]

    required_columns = [
        "age",
        "sex",
        "bmi",
        "children",
        "smoker",
        "region",
        "charges"
    ]

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        st.error(
            "The following columns are missing from your dataset: "
            + ", ".join(missing)
        )
        st.stop()

    data = data[required_columns].dropna()

    return data


df = load_dataset()

# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

X = df.drop(columns=["charges"])
y = df["charges"]

categorical_features = [
    "sex",
    "smoker",
    "region"
]

numerical_features = [
    "age",
    "bmi",
    "children"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_charges(
    age,
    sex,
    bmi,
    children,
    smoker,
    region
):

    patient = pd.DataFrame([
        {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }
    ])

    prediction = model.predict(patient)[0]

    return max(0, float(prediction))


# ============================================================
# COST CATEGORY
# ============================================================

def get_category(charges):

    if charges < 5000:
        return "Low 🟢"

    elif charges < 15000:
        return "Moderate 🟡"

    else:
        return "High 🔴"


# ============================================================
# LOGIN PAGE
# ============================================================

def login_page():

    st.markdown(
        '<div class="login-box">',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h1 style="text-align:center;color:#12355b;">
        🏥 MediCost
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style="text-align:center;color:#667085;">
        AI-Based Healthcare Cost Prediction System
        </p>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Demo Login\n\n"
        "Doctor ID: doctor\n\n"
        "Password: doctor123"
    )

    doctor_id = st.text_input(
        "👨‍⚕️ Doctor ID"
    )

    password = st.text_input(
        "🔐 Password",
        type="password"
    )

    if st.button(
        "🔓 LOGIN",
        width="stretch",
        type="primary"
    ):

        if (
            doctor_id.strip().lower() == "doctor"
            and password == "doctor123"
        ):

            st.session_state.logged_in = True
            st.session_state.doctor_name = "Dr. Anjani Priya"

            st.rerun()

        else:

            st.error(
                "Invalid Doctor ID or Password."
            )

    st.markdown(
        """
        <p style="text-align:center;color:#888;font-size:12px;">
        Academic Project / Demo Application
        </p>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# CHECK LOGIN
# ============================================================

if not st.session_state.logged_in:

    login_page()

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    "# 🏥 MediCost"
)

st.sidebar.caption(
    "Doctor Healthcare Cost Portal"
)

st.sidebar.success(
    "Logged in as " +
    st.session_state.doctor_name
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "👤 Patient Registration",
        "📋 Patient History",
        "📊 Reports & Analysis",
        "📈 Model Performance",
        "👨‍⚕️ Doctor Profile"
    ]
)

st.sidebar.markdown("---")

if st.sidebar.button(
    "🚪 Logout",
    width="stretch"
):

    st.session_state.logged_in = False
    st.session_state.doctor_name = ""

    st.rerun()


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🏠 Doctor Dashboard</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Welcome, **"
        + st.session_state.doctor_name
        + "** 👋"
    )

    total_patients = len(
        st.session_state.history
    )

    if total_patients > 0:

        average_charges = np.mean([
            record["Predicted Charges"]
            for record in st.session_state.history
        ])

        high_cost_cases = sum(
            record["Category"].startswith("High")
            for record in st.session_state.history
        )

    else:

        average_charges = 0
        high_cost_cases = 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👤 Total Patients",
        total_patients
    )

    col2.metric(
        "🔮 Predictions",
        total_patients
    )

    col3.metric(
        "💰 Average Charges",
        f"₹ {average_charges:,.0f}"
    )

    col4.metric(
        "🔴 High Cost Cases",
        high_cost_cases
    )

    st.markdown("### ⚡ Quick Information")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.info(
            "👤 **Patient Registration**\n\n"
            "Register a patient and predict healthcare charges."
        )

    with c2:

        st.info(
            "📋 **Patient History**\n\n"
            "View and download previous predictions."
        )

    with c3:

        st.info(
            "📊 **Reports**\n\n"
            "Analyze healthcare charges using charts."
        )

    st.markdown("### 📌 Dataset Information")

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "Dataset Records",
        len(df)
    )

    d2.metric(
        "Average Actual Charges",
        f"₹ {df['charges'].mean():,.0f}"
    )

    d3.metric(
        "Model R² Score",
        f"{r2:.3f}"
    )

    if st.session_state.history:

        st.markdown(
            "### 🕒 Recent Predictions"
        )

        recent = pd.DataFrame(
            st.session_state.history[-5:]
        )

        recent = recent.iloc[::-1]

        st.dataframe(
            recent,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "No patient predictions available yet."
        )


# ============================================================
# PATIENT REGISTRATION
# ============================================================

elif page == "👤 Patient Registration":

    st.markdown(
        '<div class="main-title">'
        '👤 Patient Registration & Prediction'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter patient information below. "
        "**Charges are predicted automatically by the ML model.**"
    )

    with st.form("patient_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            patient_id = st.text_input(
                "Patient ID",
                placeholder="P001"
            )

            patient_name = st.text_input(
                "Patient Name"
            )

            age = st.number_input(
                "Age",
                min_value=1,
                max_value=100,
                value=30
            )

        with col2:

            sex = st.selectbox(
                "Gender",
                [
                    "female",
                    "male"
                ]
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=60.0,
                value=25.0,
                step=0.1
            )

            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=10,
                value=0
            )

        with col3:

            smoker = st.selectbox(
                "Smoking Status",
                [
                    "no",
                    "yes"
                ]
            )

            region = st.selectbox(
                "Region",
                [
                    "southwest",
                    "southeast",
                    "northwest",
                    "northeast"
                ]
            )

            notes = st.text_area(
                "Doctor Notes"
            )

        submitted = st.form_submit_button(
            "🔮 PREDICT HEALTHCARE CHARGES",
            width="stretch",
            type="primary"
        )

    if submitted:

        if (
            not patient_id.strip()
            or not patient_name.strip()
        ):

            st.error(
                "Please enter Patient ID and Patient Name."
            )

        else:

            predicted = predict_charges(
                age,
                sex,
                bmi,
                children,
                smoker,
                region
            )

            record = {

                "Patient ID":
                    patient_id.strip(),

                "Patient Name":
                    patient_name.strip(),

                "Age":
                    age,

                "Gender":
                    sex,

                "BMI":
                    bmi,

                "Children":
                    children,

                "Smoker":
                    smoker,

                "Region":
                    region,

                "Predicted Charges":
                    round(predicted, 2),

                "Category":
                    get_category(predicted),

                "Doctor":
                    st.session_state.doctor_name,

                "Date":
                    datetime.now().strftime(
                        "%d-%m-%Y %I:%M %p"
                    ),

                "Doctor Notes":
                    notes
            }

            st.session_state.history.append(
                record
            )

            st.session_state.last_prediction = record

            st.success(
                "Patient prediction completed successfully!"
            )

    # SHOW RESULT

    if st.session_state.last_prediction:

        result = st.session_state.last_prediction

        st.markdown(
            f"""
            <div class="result-box">

            <h2>
            💰 Estimated Healthcare Charges
            </h2>

            <div class="result-money">
            ₹ {result["Predicted Charges"]:,.2f}
            </div>

            <h3>
            {result["Category"]}
            </h3>

            <p>
            <b>Patient:</b>
            {result["Patient Name"]}
            </p>

            <p>
            <b>Patient ID:</b>
            {result["Patient ID"]}
            </p>

            <p>
            <b>Age:</b> {result["Age"]}
            &nbsp;&nbsp;
            <b>BMI:</b> {result["BMI"]}
            &nbsp;&nbsp;
            <b>Smoker:</b> {result["Smoker"]}
            </p>

            <p style="color:#667085;font-size:13px;">
            This is an ML-based academic estimate and
            not a guaranteed medical bill.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PATIENT HISTORY
# ============================================================

elif page == "📋 Patient History":

    st.markdown(
        '<div class="main-title">📋 Patient History</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.history:

        st.info(
            "No patient records available."
        )

    else:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        search = st.text_input(
            "🔎 Search Patient ID or Patient Name"
        )

        if search.strip():

            search_text = search.lower().strip()

            history_df = history_df[
                history_df["Patient ID"]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_text,
                    na=False
                )
                |
                history_df["Patient Name"]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_text,
                    na=False
                )
            ]

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True
        )

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Patient History",
            csv_data,
            "patient_history.csv",
            "text/csv",
            width="stretch"
        )


# ============================================================
# REPORTS & ANALYSIS
# ============================================================

elif page == "📊 Reports & Analysis":

    st.markdown(
        '<div class="main-title">'
        '📊 Reports & Healthcare Cost Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Dataset Records",
        len(df)
    )

    col2.metric(
        "Average Charges",
        f"₹ {df['charges'].mean():,.0f}"
    )

    col3.metric(
        "Maximum Charges",
        f"₹ {df['charges'].max():,.0f}"
    )

    col4.metric(
        "Minimum Charges",
        f"₹ {df['charges'].min():,.0f}"
    )

    # AGE VS CHARGES

    st.markdown(
        "### 📈 Age vs Healthcare Charges"
    )

    fig1 = px.scatter(
        df,
        x="age",
        y="charges",
        color="smoker",
        title="Age vs Healthcare Charges"
    )

    st.plotly_chart(
        fig1,
        width="stretch"
    )

    # BMI VS CHARGES

    st.markdown(
        "### 📈 BMI vs Healthcare Charges"
    )

    fig2 = px.scatter(
        df,
        x="bmi",
        y="charges",
        color="smoker",
        title="BMI vs Healthcare Charges"
    )

    st.plotly_chart(
        fig2,
        width="stretch"
    )

    # SMOKER ANALYSIS

    st.markdown(
        "### 🚬 Smoker vs Non-Smoker"
    )

    smoker_data = (
        df.groupby("smoker")["charges"]
        .mean()
        .reset_index()
    )

    fig3 = px.bar(
        smoker_data,
        x="smoker",
        y="charges",
        title="Average Charges by Smoking Status"
    )

    st.plotly_chart(
        fig3,
        width="stretch"
    )

    # REGION ANALYSIS

    st.markdown(
        "### 🌍 Charges by Region"
    )

    region_data = (
        df.groupby("region")["charges"]
        .mean()
        .reset_index()
    )

    fig4 = px.bar(
        region_data,
        x="region",
        y="charges",
        title="Average Charges by Region"
    )

    st.plotly_chart(
        fig4,
        width="stretch"
    )

    # PATIENT PREDICTIONS

    if st.session_state.history:

        st.markdown(
            "### 👥 Registered Patient Predictions"
        )

        patient_df = pd.DataFrame(
            st.session_state.history
        )

        fig5 = px.bar(
            patient_df,
            x="Patient Name",
            y="Predicted Charges",
            color="Category",
            title="Predicted Charges of Patients"
        )

        st.plotly_chart(
            fig5,
            width="stretch"
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📈 Model Performance":

    st.markdown(
        '<div class="main-title">'
        '📈 Machine Learning Model Performance'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "R² Score",
        f"{r2:.3f}"
    )

    col2.metric(
        "MAE",
        f"₹ {mae:,.2f}"
    )

    col3.metric(
        "RMSE",
        f"₹ {rmse:,.2f}"
    )

    st.markdown(
        "### 🎯 Actual vs Predicted Charges"
    )

    comparison_df = pd.DataFrame({

        "Actual Charges":
            y_test.values,

        "Predicted Charges":
            y_pred

    })

    fig = px.scatter(
        comparison_df,
        x="Actual Charges",
        y="Predicted Charges",
        title="Actual vs Predicted Healthcare Charges"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.markdown(
        "### 📋 Prediction Comparison"
    )

    st.dataframe(
        comparison_df,
        width="stretch",
        hide_index=True
    )

    st.markdown(
        "### 🧠 Model Input Variables"
    )

    st.write(
        """
        The model uses the following patient characteristics:

        • Age  
        • Gender  
        • BMI  
        • Number of Children  
        • Smoking Status  
        • Region  

        The target/output variable is **charges**.
        """
    )


# ============================================================
# DOCTOR PROFILE
# ============================================================

elif page == "👨‍⚕️ Doctor Profile":

    st.markdown(
        '<div class="main-title">👨‍⚕️ Doctor Profile</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="card">

        <h2>
        👨‍⚕️ {st.session_state.doctor_name}
        </h2>

        <p>
        <b>Doctor ID:</b> doctor
        </p>

        <p>
        <b>Specialization:</b> General Medicine
        </p>

        <p>
        <b>Hospital:</b> MediCost Healthcare Center
        </p>

        <p>
        <b>System:</b> AI-Based Healthcare Cost Prediction
        </p>

        <p>
        <b>Total Predictions:</b>
        {len(st.session_state.history)}
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Demo Login: Doctor ID = doctor | Password = doctor123"
    )

    st.warning(
        "This is an academic project. "
        "Real healthcare deployment would require "
        "secure authentication, encrypted databases, "
        "privacy controls, audit logs and appropriate "
        "healthcare compliance."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    <b>🏥 MediCost</b><br>

    AI-Based Healthcare Cost Prediction &
    Doctor Management System<br>

    Python • Streamlit • Scikit-learn • Plotly

    </div>
    """,
    unsafe_allow_html=True
)
