from __future__ import annotations

from pathlib import Path

from dsg_spatialqa_lab import load_dashboard_bundle


STREAMLIT_MISSING_MESSAGE = (
    "Streamlit is not installed. Install dashboard extras before running this app."
)


def main() -> int:
    try:
        import streamlit as st
    except ImportError:
        print(STREAMLIT_MISSING_MESSAGE)
        return 1

    bundle_path = Path("dashboard.json")
    if not bundle_path.exists():
        st.error("dashboard.json not found in the current directory.")
        return 1
    bundle = load_dashboard_bundle(bundle_path)
    st.title("DSG-SpatialQA Dashboard")
    st.json(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
