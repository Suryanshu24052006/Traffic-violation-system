"""
dashboard/app.py
===================
Streamlit dashboard for the traffic-violation system:
  - Trajectory heatmap + latest annotated video output
  - Live violation log (from detected_violations table)
  - Per-vehicle risk profile lookup by plate number
  - Per-driver risk profile lookup by license number (aggregated across
    every vehicle registered to that person)
  - Municipal accident-hotspot alerts (per-camera safety priority + report)

Run:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from database.db_manager import DBManager  # noqa: E402
from src.risk.risk_engine import RiskEngine  # noqa: E402
from src.hotspot.hotspot_analyzer import HotspotAnalyzer  # noqa: E402

st.set_page_config(page_title="Traffic Violation & Risk Profiling System", layout="wide")


@st.cache_resource
def load_config():
    with open(Path(__file__).parent.parent / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


config = load_config()
db = DBManager(config["database"]["path"])
risk_engine = RiskEngine(config)
hotspot_analyzer = HotspotAnalyzer(config, db)

st.title("🚦 Smart Traffic Violation & Risk Profiling System")
st.caption(
    "Vehicle detection & tracking · trajectory heatmaps · helmet / speed / "
    "truck-hours violations · ANPR · database-driven risk scoring · "
    "municipal accident-hotspot alerts. Database is synthetic demo data — see README for why."
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📹 Run Output", "📋 Violation Log", "🔍 Vehicle Risk Lookup",
    "🪪 Driver Risk Lookup", "🚧 Municipal Hotspot Alerts",
])

# --------------------------------------------------------------------- #
with tab1:
    col1, col2 = st.columns(2)
    outputs_dir = Path(config["paths"]["outputs_dir"])

    with col1:
        st.subheader("Trajectory Heatmap")
        heatmap_path = outputs_dir / "trajectory_heatmap.jpg"
        if heatmap_path.exists():
            st.image(str(heatmap_path), use_container_width=True)
        else:
            st.info("No heatmap yet — run the pipeline first:\n\n"
                    "`python -m src.pipeline.main_pipeline --video <path>`")

    with col2:
        st.subheader("Annotated Output Video")
        video_path = outputs_dir / "annotated_output.mp4"
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.info("No annotated video yet — run the pipeline first.")

# --------------------------------------------------------------------- #
with tab2:
    st.subheader("Detected Violations (this system's own live log)")
    violations = db.get_recent_detected_violations(limit=200)
    if violations:
        df = pd.DataFrame(violations)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Violations by type")
        st.bar_chart(df["violation_type"].value_counts())
    else:
        st.info("No violations logged yet. Run the pipeline on a video to populate this.")

# --------------------------------------------------------------------- #
with tab3:
    st.subheader("Look up a vehicle's risk profile")
    all_plates = db.get_all_plate_numbers()
    plate = st.selectbox(
        "Select a plate number (from the mock database)", options=[""] + sorted(all_plates)
    )
    manual_plate = st.text_input("...or type any plate number directly")
    lookup_plate = manual_plate.strip().upper() or plate

    if lookup_plate:
        profile = db.get_vehicle_profile(lookup_plate)
        result = risk_engine.compute_risk_score(profile)

        category_color = {
            "low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴",
        }.get(result.risk_category, "⚪")

        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Category", f"{category_color} {result.risk_category.upper()}")
        c2.metric("Risk Score", f"{result.risk_score:.2f}")
        c3.metric("Found in DB", "Yes" if profile.found else "No")

        if profile.found:
            st.write(f"**Owner:** {profile.owner_name} (license: `{profile.owner_license_number}`)  "
                     f"|  **Type:** {profile.vehicle_type}  |  **City:** {profile.city}, {profile.state}")
            st.write(f"**Challans:** {profile.challan_count} total, "
                     f"{profile.unpaid_challan_count} unpaid "
                     f"(₹{profile.total_unpaid_fine} outstanding)")
            st.write(f"**Intoxication records:** {profile.intoxication_count}")
            st.write(f"**Open hit-and-run cases:** {profile.open_hit_and_run_count}")
            st.caption(
                "💡 This owner may have other vehicles too — see the Driver Risk Lookup "
                "tab for the aggregated view across everything registered to their license."
            )

            if profile.challan_history:
                st.subheader("Challan history")
                st.dataframe(pd.DataFrame(profile.challan_history), use_container_width=True, hide_index=True)

        st.subheader("Why this score?")
        for line in result.explanation:
            st.write(f"- {line}")

# --------------------------------------------------------------------- #
with tab4:
    st.subheader("Look up a driver's aggregated risk profile")
    st.caption(
        "Keyed by license number, not name — see schema.sql for why name-matching "
        "isn't reliable (two people can share a name; a license number is unique)."
    )
    all_licenses = db.get_all_license_numbers()
    license_choice = st.selectbox(
        "Select a license number (from the mock database)", options=[""] + sorted(all_licenses)
    )
    manual_license = st.text_input("...or type any license number directly")
    lookup_license = manual_license.strip().upper() or license_choice

    if lookup_license:
        driver_profile = db.get_owner_profile(lookup_license)
        driver_result = risk_engine.compute_driver_risk_score(driver_profile)

        category_color = {
            "low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴",
        }.get(driver_result.risk_category, "⚪")

        c1, c2, c3 = st.columns(3)
        c1.metric("Driver Risk Category", f"{category_color} {driver_result.risk_category.upper()}")
        c2.metric("Driver Risk Score", f"{driver_result.risk_score:.2f}")
        c3.metric("Vehicles Registered", driver_profile.vehicle_count if driver_profile.found else 0)

        if driver_profile.found:
            st.write(f"**Name:** {driver_profile.name}")
            st.write(f"**Registered vehicles:** {', '.join(driver_profile.plate_numbers) or 'none'}")
            st.write(f"**Total challans (all vehicles):** {driver_profile.total_challan_count} "
                     f"({driver_profile.total_unpaid_challan_count} unpaid, "
                     f"₹{driver_profile.total_unpaid_fine} outstanding)")
            st.write(f"**Intoxication records:** {driver_profile.intoxication_count}")
            st.write(f"**Open hit-and-run cases:** {driver_profile.open_hit_and_run_count}")

        st.subheader("Why this score?")
        for line in driver_result.explanation:
            st.write(f"- {line}")

# --------------------------------------------------------------------- #
with tab5:
    st.subheader("Municipal accident-hotspot alerts")
    st.caption(
        "Aggregates recurring violations per camera location, cross-referenced against "
        "known accident black-spot history, into a safety-priority score and recommended "
        "action — see src/hotspot/hotspot_analyzer.py."
    )

    camera_ids = db.get_all_camera_ids()
    colA, colB = st.columns([1, 3])
    with colA:
        selected_camera = st.selectbox("Camera", options=camera_ids)
        lookback = st.number_input("Lookback window (days)", min_value=1, max_value=365,
                                     value=config["municipal"]["lookback_days"])
        run_btn = st.button("Generate / refresh report", type="primary")

    with colB:
        if run_btn and selected_camera:
            report = hotspot_analyzer.analyze_camera(selected_camera, lookback_days=int(lookback))
            priority_color = {
                "low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴",
            }.get(report.priority_level, "⚪")

            st.metric("Priority Level", f"{priority_color} {report.priority_level.upper()}")
            st.metric("Safety Priority Score", f"{report.safety_priority_score:.2f}")
            st.write(f"**Location:** {report.location_name}")
            st.write(f"**Violations in window:** {report.violation_count} "
                     f"(dominant type: {report.dominant_violation_type or 'none'})")
            st.write(f"**Recommended action:** {report.recommended_action}")

            if report.violation_breakdown:
                st.bar_chart(pd.Series(report.violation_breakdown))

            if report.heatmap_path and Path(report.heatmap_path).exists():
                st.image(report.heatmap_path, caption="Violation-density heatmap for this location",
                          use_container_width=True)

            st.subheader("Why this priority?")
            for line in report.explanation:
                st.write(f"- {line}")

    st.divider()
    st.subheader("Known accident black spots (municipal register)")
    black_spots = db.get_all_black_spots()
    if black_spots:
        st.dataframe(pd.DataFrame(black_spots), use_container_width=True, hide_index=True)
    else:
        st.info("No black spots registered.")

    st.subheader("Recent municipal alerts generated")
    alerts = db.get_recent_municipal_alerts(limit=20)
    if alerts:
        st.dataframe(pd.DataFrame(alerts), use_container_width=True, hide_index=True)
    else:
        st.info("No municipal alerts generated yet — run the pipeline with --camera-id, "
                "or generate one above.")
