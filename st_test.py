import streamlit as st
import pandas as pd

st.write("FIRST ATTEMPT ARE YOU SERIOUS RIGHT NEOW")
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40],
    'third column': [10, 20, 30, 40]
})

st.write(df)