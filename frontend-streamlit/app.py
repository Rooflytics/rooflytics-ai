import streamlit as st
import requests
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

def load_tif_for_display(path):
    with rasterio.open(path) as src:
        data = src.read()

    if data.shape[0] == 3:
        # RGB
        img = np.transpose(data, (1, 2, 0))
        img = np.clip(img / 255.0, 0, 1)
        return img
    else:
        # single band
        return data[0]

def overlay_thermal_mask(rgb, thermal_mask):
    overlay = rgb.copy()

    # Hot roofs → red
    overlay[thermal_mask == 1] = [1.0, 0.0, 0.0]

    # Cool roofs → blue
    overlay[thermal_mask == 2] = [0.0, 0.4, 1.0]

    alpha = 0.5
    blended = (1 - alpha) * rgb + alpha * overlay

    return blended


API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Rooflytics", layout="centered")

st.title("🏠 Rooflytics")
st.subheader("Urban Roof Intelligence & Energy Impact")

uploaded_file = st.file_uploader(
    "Upload GeoTIFF (.tif)",
    type=["tif"],
)

if uploaded_file:
    st.success("File uploaded")

    if st.button("Run Analysis"):
        with st.spinner("Uploading file..."):
            files = {"file": uploaded_file}
            r = requests.post(f"{API_BASE}/upload", files=files)
            job_id = r.json()["job_id"]

        with st.spinner("Running roof analysis..."):
            r = requests.post(f"{API_BASE}/process/{job_id}")
            if r.status_code != 200:
                st.error("Backend error during processing")
                st.code(r.text)
                st.stop()

            result = r.json()

        st.success("Analysis complete")

        st.markdown("### Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Roofs", result["num_roofs"])
        col2.metric("Cool Roofs", result["cool_roofs"])
        col3.metric("Hot Roofs", result["hot_roofs"])

        st.markdown("### Environmental Impact")

        col4, col5 = st.columns(2)

        col4.metric(
            "Energy Savings (kWh / year)",
            f'{result["total_energy_kwh_per_year"]:,}',
        )

        col5.metric(
            "CO₂ Reduction (kg / year)",
            f'{result["total_co2_kg_per_year"]:,}',
        )

        st.markdown("### Economic Impact")

        st.metric(
            "Estimated Cost Savings (NZD / year)",
            f'NZD {result["total_cost_nzd_per_year"]:,}',
        )


        st.markdown("### Download Results")

        files = requests.get(
            f"{API_BASE}/results/{job_id}"
        ).json()["files"]

        for f in files:
            url = f"{API_BASE}/results/{job_id}/{f}"
            st.markdown(f"- [{f}]({url})")
        
        st.markdown("### 🗺️ Thermal Roof Visualization")

        job_results_url = f"{API_BASE}/results/{job_id}"

        # Load images
        input_tif = requests.get(
            f"{job_results_url}/input.tif"
        ).content

        thermal_tif = requests.get(
            f"{job_results_url}/thermal_clusters.tif"
        ).content

        # Save temporarily
        with open("temp_input.tif", "wb") as f:
            f.write(input_tif)

        with open("temp_thermal.tif", "wb") as f:
            f.write(thermal_tif)

        rgb = load_tif_for_display("temp_input.tif")
        thermal = load_tif_for_display("temp_thermal.tif")

        overlay = overlay_thermal_mask(rgb, thermal)

        st.image(
            overlay,
            caption="Red = Hot roofs | Blue = Cool roofs",
            width="stretch",
        )