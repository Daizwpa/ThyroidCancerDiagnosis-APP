
import pandas as pd
import streamlit as st
from PIL import Image
import io
from Controller.acr_Ti_rads import acr_ti_rads
from Controller.Classify import Classifier


def Page(dataset: pd.DataFrame = None):
    st.header(" Classification tumour:", divider='rainbow')
    st.markdown("### Input Ultrasound Image and Mask:")
    col1, col2, = st.columns(2)

    with col1:
        uploaded_Image = st.file_uploader("Choose a Ultrasound image", type=[
            "png", "jpg", "dcm", "nii"], accept_multiple_files=False)
        image = None
        if uploaded_Image is not None:
            bytes = uploaded_Image.read()
            image = Image.open(io.BytesIO(bytes))
            st.image(image)
        else:
            st.image("https://img.lovepik.com/element/40217/7532.png_1200.png")
    with col2:
        uploaded_Mask = st.file_uploader("Choos a Mask of the image", type=[
            "png", "jpg", "dcm", "nii"], accept_multiple_files=False)
        mask = None
        if uploaded_Mask is not None:
            bytes = uploaded_Mask.read()
            mask = Image.open(io.BytesIO(bytes))
            st.image(mask)
        else:
            st.image("https://img.lovepik.com/element/40217/7532.png_1200.png")
    st.divider()
    st.markdown("### Input Clinical data:")

    ClinicalColn1, ClinicalColn2 = st.columns(2)

    with ClinicalColn1:
        age_box = st.text_input("Age")
    with ClinicalColn2:
        sex_box = st.option = st.selectbox("Sex", ("Female", "Male"))

    sizeColn1, sizeColn2, sizeColn3 = st.columns(3)

    with sizeColn1:
        size_x_box = st.text_input("Size X")
    with sizeColn2:
        size_y_box = st.text_input("Size Y")
    with sizeColn3:
        size_z_box = st.text_input("Size Z")

    shape_box = st.selectbox("Ti-rads shape", acr_ti_rads["SHAPE"].keys())
    margin_box = st.selectbox("Ti-rads margin", acr_ti_rads["MARGIN"].keys())
    foci_box = st.multiselect("Ti-rads echogenicfoci",
                              acr_ti_rads["ECHOGENIC FOCI"].keys())
    level_box = st.selectbox(
        "Ti-rads level", acr_ti_rads["TI-RADS Level"].keys())

    st.divider()

    clinical_data = None
    result = None
    if st.button("Classify", type="primary", use_container_width=True):
        clinical_data = {
            "age": [int(age_box)],
            "sex": [sex_box],
            "size_x": [float(size_x_box)],
            "size_y": [float(size_y_box)],
            "size_z": [float(size_z_box)],
            "ti-rads_shape": [acr_ti_rads["SHAPE"][shape_box]],
            "ti-rads_margin": [acr_ti_rads["MARGIN"][margin_box]],
            "ti-rads_echogenicfoci": [sum([acr_ti_rads["ECHOGENIC FOCI"][y] for y in foci_box])],
            "ti-rads_level": [acr_ti_rads["TI-RADS Level"][level_box]]
        }
        df = pd.DataFrame(clinical_data)
        c = Classifier(
            wieghts_path="./assets/Trained Model/1model0.keras",
            pipeline_sclaer_path="./assets/preprocess.obj",
            logPathRadimoics="./Logs/RadiomicsLog.txt",
            SettingsPathRadimoics="./settings/radiomics.yaml"
        )
        result = c.Classify(
            image=image,
            mask=mask,
            clinical_data=df
        )

    st.divider()
    st.markdown("### Result:")
    if (result is not None):
        st.write(result)


Page(pd.DataFrame({}))
