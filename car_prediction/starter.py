import streamlit as st
import numpy as np
from joblib import load
import pandas as pd

# Initialize the session state with an empty prediction
if "pred" not in st.session_state:
    st.session_state["pred"] = None
    #st.session_state stores value across user interactions

# Function to load the model (use caching)
@st.cache_resource(show_spinner="Loading Model...")
def load_model():
    try:
        return load("model/model.joblib")
    except ModuleNotFoundError as e:
        st.error(
            "Failed to load the model: missing module required by the trained pipeline.\n\n"
            f"Missing: `{e.name}`\n\n"
            "Fix: install the scikit-learn version used for training (see warnings "
            "above), or add a local `_loss.py` shim that proxies to `sklearn._loss`, "
            "or re-export the model with `skops`."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load the model:\n\n{e}")
        st.stop()

# Callback function to make a prediction
def make_prediction(pipe):
    X_pred=pd.DataFrame([{
        "miles": st.session_state["miles"],
        "year":st.session_state["year"],
        "make":st.session_state["make"],
        "model":st.session_state["model"],
        "body_type":st.session_state["body_type"],
        "vehicle_type":st.session_state["vehicle_type"],
        "drivetrain":st.session_state["drivetrain"],
        "transmission":st.session_state["transmission"],
        "fuel_type":st.session_state["fuel_type"],
        "engine_size":st.session_state["engine_size"],

        "engine_block":st.session_state["engine_block"],
        "state":st.session_state["province"]

    }])
    pred=pipe.predict(X_pred)
    st.session_state["pred"] = round(pred[0], 2)

if __name__ == "__main__":
    st.title("Prediction Car Price")

    # Load model
    pipe = load_model()

    with st.form(key="form"):
        col1, col2, col3 = st.columns(3)

        # Column 1 inputs
        with col1:
            st.number_input("Miles", value=86132.0, min_value=0.0, step=0.1, key="miles")
            st.selectbox("Model", index=0, key="model", options=[
                'Prius', 'Highlander', 'Civic', 'Accord', 'Corolla', 'Ridgeline',
                'Odyssey', 'CR-V', 'Pilot', 'Camry Solara', 'Matrix', 'RAV4',
                'Rav4', 'HR-V', 'Fit', 'Yaris', 'Yaris iA', 'Tacoma', 'Camry',
                'Avalon', 'Venza', 'Sienna', 'Passport', 'Accord Crosstour',
                'Crosstour', 'Element', 'Tundra', 'Sequoia', 'Corolla Hatchback',
                '4Runner', 'Echo', 'Tercel', 'MR2 Spyder', 'FJ Cruiser',
                'Corolla iM', 'C-HR', 'Civic Hatchback', '86', 'S2000', 'Supra',
                'Insight', 'Clarity', 'CR-Z', 'Prius Prime', 'Prius Plug-In',
                'Prius c', 'Prius C', 'Prius v'
            ])
            st.selectbox("Body Type", key="body_type", options=[
                "suv", "sedan", "pickup", "hatchback", "minivan", "coupe", "wagon",
                "convertible", "cargo_van", "mini_mpv", "passenger_van", "cutaway",
                "targa", "micro_car", "car_van", "chassis_cab", "crossover",
                "combi", "roadster", "commercial_wagon"
            ])
            st.selectbox("Vehicle Type", key="vehicle_type", options=["Car", "Truck"])

        # Column 2 inputs
        with col2:
            st.number_input("Year", value=2001, min_value=1886, step=1, key="year")
            st.number_input("Engine size (L)", value=1.5, key="engine_size", min_value=0.9, step=0.1)
            st.selectbox("Drivetrain", ["FWD", "RWD", "4WD"], key="drivetrain")
            st.selectbox("Transmission", ["Automatic", "Manual"], key="transmission")

        # Column 3 inputs
        with col3:
            st.selectbox("Make", key="make", index=0, options=['toyota', 'honda'])
            st.selectbox("Province", index=0, key="province", options=[
                'NB', 'QC', 'BC', 'ON', 'AB', 'MB', 'SK', 'NS',
                'PE', 'NL', 'YT', 'NC', 'OH', 'SC'
            ])
            st.selectbox("Fuel Type", ["gasoline", "hyrid", "diesel"], key="fuel_type")  # keep typo 'hyrid'
            st.selectbox("Engine Block", ["I", "V", "H"], key="engine_block")
        #Submit button
        st.form_submit_button("Calculate", type="primary", on_click=make_prediction, kwargs=dict(pipe=pipe))

    # Display the prediction
    if st.session_state["pred"] is not None:
        st.subheader(f"Predicted Price of Reused Car: ${st.session_state['pred']}")
    else:
        st.write("Calculate to get predicted price")

    # Debug: show all state values
    st.write(st.session_state)
