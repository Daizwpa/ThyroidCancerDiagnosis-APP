import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.calculation.calc import Get_DataFrame_Info, Get_DataFrame_Value_info
from utils.calculation.chart import Get_barPlot, Get_Density_Destributation, Get_Box_Plot, Get_HistogramPlot
import os
from utils.core import Load_data


def Page(dataset):
    st.title(os.getenv('TITLE_APP'))
    st.write("The whole cheatsheet:")

    st.write(dataset)

    st.write("Some statistics:")
    st.write(dataset.describe())

    st.divider()
    st.write("Some information:")
    st.table(Get_DataFrame_Info(dataset))

    st.divider()
    st.header("Column type:")
    typeOfFeatureDic = {x: eval(os.getenv(x))
                        for x in eval(os.getenv('COLUMN_TYPE'))}
    tabsOfFeatursTypes = zip(st.tabs(list(typeOfFeatureDic)), typeOfFeatureDic)
    for tab, typeofFeature in tabsOfFeatursTypes:
        with tab:
            tab.header(f'List of {typeofFeature} columns')
            st.table(Get_DataFrame_Value_info(
                dataset, typeOfFeatureDic[typeofFeature], typeofFeature))

    st.divider()
    st.header("BarPlot:")
    listOfcolumnForHistogram = eval(os.getenv("BarPlot"))
    listoftab = zip(st.tabs(listOfcolumnForHistogram),
                    listOfcolumnForHistogram)
    for tab, featureName in listoftab:
        with tab:
            tab.header(f'{featureName}')
            fig, ax = Get_barPlot(dataset, featureName)
            st.pyplot(fig)

    st.divider()
    st.header('Scatter Plot:')

    options = st.multiselect(
        'select two columns for makeing correlation plot',
        list(dataset.columns.values),
        max_selections=2
    )

    if len(options) == 2:
        st.scatter_chart(dataset, x=options[0], y=options[1])

    st.divider()

    st.header("Density plot:")
    optionDensity = st.selectbox(
        'What would you plot?',
        eval(os.getenv("DensityPlot")), key="1")
    if len(optionDensity):
        fig, ax = Get_Density_Destributation(dataset, optionDensity)
        st.pyplot(fig)

    st.divider()

    st.header("Histogram plot:")
    optionHistogram = st.selectbox(
        'What would you plot?',
        eval(os.getenv("DensityPlot")),  key="<uniquevalueofsomesort>")
    if len(optionHistogram):
        fig, ax = Get_HistogramPlot(dataset, optionHistogram)
        st.pyplot(fig)

    st.divider()

    st.header("Box plot:")
    optionBoxPlot = st.selectbox(
        'What would you plot?', eval(os.getenv("Quantitative")), key="3")
    if len(optionBoxPlot):
        # fig, ax = Get_Box_Plot(dataset, optionBoxPlot)
        fig = Get_Box_Plot(dataset, optionBoxPlot)
        st.pyplot(fig)

    pass


Page(Load_data(os.getenv("DATASET_CSV"), index_col="annot_id"))
