import streamlit as st
import numpy as np
import pandas as pd
import joblib
import requests
import polyline
import folium
from streamlit_folium import st_folium

# ---------------------- APP CONFIGURATION ----------------------

st.set_page_config(page_title="CO₂ Route Optimizer", layout="centered")
st.title("🚚 CO₂ Emission Predictor for Logistics Routes")

# Session state initialization
if 'best_route' not in st.session_state:
    st.session_state.best_route = None
    st.session_state.prediction_done = False

# ---------------------- CACHED FUNCTIONS ----------------------

@st.cache_resource
def load_model():
    model_path = r"E:\ICBP 2.0\FINAL PROJECT\model_pipeline_carbon_footprint_route_optimizer.pkl"
    return joblib.load(model_path)

@st.cache_data
def load_route_data():
    data_path = r"E:\ICBP 2.0\FINAL PROJECT\all_route_variants_with_emissions.csv"
    return pd.read_csv(data_path)

model = load_model()
df = load_route_data()

# ---------------------- GOOGLE MAPS API ----------------------

API_KEY = "AIzaSyCX1lL08h42xp7vq57_09fW84slw1B-g6o"  

def get_route_coordinates(origin, destination):
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin}&destination={destination}&key={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'OK':
        encoded_polyline = data['routes'][0]['overview_polyline']['points']
        return polyline.decode(encoded_polyline)
    return []

def plot_route_on_map(route_coordinates):
    if not route_coordinates:
        st.warning("No route coordinates available.")
        return

    start = route_coordinates[0]
    end = route_coordinates[-1]

    route_map = folium.Map(location=start, zoom_start=10)
    folium.PolyLine(route_coordinates, color='blue', weight=4).add_to(route_map)
    folium.Marker(start, popup="Origin", icon=folium.Icon(color='green')).add_to(route_map)
    folium.Marker(end, popup="Destination", icon=folium.Icon(color='red')).add_to(route_map)
    st_folium(route_map, width=700)

# ---------------------- PREDICTION FUNCTION ----------------------

def generate_routes_and_predict(user_input, model, route_data):
    origin = user_input['Origin']
    destination = user_input['Destination']
    traffic = user_input['Traffic']
    weather = user_input['Weather']
    cargo_weight = user_input['Cargo_Weight_kg']
    fuel_efficiency = user_input['Fuel_Efficiency_kmpl']

    possible_routes = []
    unique_route_summaries = route_data['Route_Summary'].unique()

    for summary in unique_route_summaries:
        route_subset = route_data[route_data['Route_Summary'] == summary]
        if route_subset.empty:
            continue
        selected_route = route_subset.iloc[0]
        distance_km = selected_route['Route_Distance_km']
        route_type = selected_route['Route_Type']
        fuel_used = distance_km / fuel_efficiency

        features = {
            'Route_Distance_km': distance_km,
            'Route_Type': route_type,
            'Traffic': traffic,
            'Weather': weather,
            'Cargo_Weight_kg': cargo_weight,
            'Fuel_Efficiency_kmpl': fuel_efficiency,
            'Adjusted_Efficiency': fuel_efficiency,
            'Fuel_Used_Litres': fuel_used
        }

        input_df = pd.DataFrame([features])
        predicted_emission = model.predict(input_df)[0]

        possible_routes.append({
            'Route ID': f"{origin}-{destination}-{summary}",
            'Predicted_CO2_Emissions': predicted_emission
        })

    return possible_routes

# ---------------------- STREAMLIT FORM ----------------------

with st.form("user_input_form"):
    origin = st.text_input("Origin City", value="Ahmedabad")
    destination = st.text_input("Destination City", value="Surat")
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High"])
    weather = st.selectbox("Weather", ["Clear", "Rainy", "Foggy", "Summer", "Storm"])
    cargo_weight = st.number_input("Cargo Weight (kg)", min_value=100.0, value=500.0)
    fuel_efficiency = st.number_input("Fuel Efficiency (km/l)", min_value=1.0, value=12.0)
    submit = st.form_submit_button("Predict Best Route")

# ---------------------- PREDICTION LOGIC ----------------------

if submit:
    user_input = {
        'Origin': origin,
        'Destination': destination,
        'Traffic': traffic,
        'Weather': weather,
        'Cargo_Weight_kg': cargo_weight,
        'Fuel_Efficiency_kmpl': fuel_efficiency
    }

    st.info("Calculating emissions for all route options...")
    predictions = generate_routes_and_predict(user_input, model, df)

    if predictions:
        best = min(predictions, key=lambda x: x['Predicted_CO2_Emissions'])
        st.session_state.best_route = best
        st.session_state.prediction_done = True
    else:
        st.error("No valid routes found.")
        st.session_state.best_route = None
        st.session_state.prediction_done = False

# ---------------------- DISPLAY RESULTS ----------------------

if st.session_state.prediction_done and st.session_state.best_route:
    best = st.session_state.best_route
    st.success(f"✅ Best Route: {best['Route ID']}")
    st.metric("Predicted CO₂ Emission (kg)", round(best['Predicted_CO2_Emissions'], 2))

    st.subheader("🗺️ Visualize Best Route on Map")
    coords = get_route_coordinates(origin, destination)
    plot_route_on_map(coords)

# ---------------------- RESET BUTTON ----------------------

if st.button("🔄 Clear Results"):
    st.session_state.best_route = None
    st.session_state.prediction_done = False
    st.experimental_rerun()
