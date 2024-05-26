import streamlit as st
import pandas as pd
from personal_ai import *

st.set_page_config(layout="wide")

personal_ai = PersonalAI()
personal_ai.run()

placeholder = st.empty()
while True:
    frame, results = personal_ai.image_q.get()

    with placeholder.container():
        st.image(frame)