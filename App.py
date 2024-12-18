from dotenv import load_dotenv
import os
import streamlit as st


def main():
    # config_sidebar()
    st.markdown("# Welcome")
    st.markdown(
        "**Web application for diagnosis thyroid cancer based on fusion data**")


if __name__ == '__main__':
    load_dotenv(override=True)
    main()
