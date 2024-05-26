import cv2
import streamlit as st
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2
import queue
import pandas as pd
from personal_ai import *
from time import time, sleep

st.set_page_config(layout="wide")
personalAI = PersonalAI("src/push-up.mp4")
personalAI.run()

st.sidebar.title("AI Personal Trainer")
display_charts = st.sidebar.checkbox('Display charts', value=True)
reset = st.sidebar.button("Reset")
st.sidebar.info("Developed by [Guilherme Perroni](http://guilhermeperroni.com)")

col1, col2 = st.columns(2)

frame, landmarks, ts = personalAI.image_q.get()
df_nodes_x = pd.DataFrame()
df_nodes_y = pd.DataFrame()

placeholder = st.empty()
c = 0
status = "Idle"
useful_points = [11, 12, 24, 23, 14, 13]

count = 0

while True:
    frame, landmarks, ts = personalAI.image_q.get()
    if ts == "done": break

    if len(landmarks.pose_landmarks) > 0:
        frame, elbow_angle = personalAI.draw_angle(frame, landmarks, 12, 14, 16)
        frame, hip_angle = personalAI.draw_angle(frame, landmarks, 11, 23, 25)

        df_y = pd.DataFrame([i.y for i in [i for i in landmarks.pose_landmarks[0]]]).rename(columns={0: ts}).transpose()
        df_nodes_y = pd.concat([df_nodes_y, df_y])
        
        # Pushup Logic
        if elbow_angle > 150 and hip_angle > 170:
            status = "Ready"
            dir = "Down"

        if status == "Ready":
            if dir == "Down" and elbow_angle < 60:
                dir = "Up"
                count += 0.5
            if dir == "Up" and elbow_angle > 100:
                dir = "Down"
                count += 0.5

        with placeholder.container():
            col1, col2 = st.columns([0.4, 0.6])
            status_m = f":red[{status}]" if status == "Idle" else f":green[{status}]"
            col2.markdown(f"### **Status:** {status_m}")
            col2.markdown(f"### Count: {int(count)}")

            col2.divider()
            col1.image(frame)
            
            if c % 2 == 0 and display_charts:
                col2.line_chart(df_nodes_y[useful_points])
            c += 1 