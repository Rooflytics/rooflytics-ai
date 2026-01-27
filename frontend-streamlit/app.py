import streamlit as st
import requests

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
