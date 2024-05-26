import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import threading
import queue


class PersonalAI:
    def __init__(self, file_name="src/push-up.mp4"):
        self.file_name = file_name
        self.model_path = "pose_landmarker_full.task"
        self.image_q = queue.Queue()
        self.options = python.vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=self.model_path), running_mode=python.vision.RunningMode.VIDEO)

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image

    def process_video(self, draw, display):
        with python.vision.PoseLandmarker.create_from_options(self.options) as landmarker:
            cap = cv2.VideoCapture(self.file_name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            calc_ts = 0

            while cap.isOpened():
                ret, frame = cap.read()

                if ret == True:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    calc_ts = int(calc_ts + 1000 / fps)
                    detection_result = landmarker.detect_for_video(mp_image, calc_ts)
                    if draw:
                        frame = self.draw_landmarks_on_image(frame, detection_result)

                    if display:
                        cv2.imshow("Frame", frame)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.image_q.put((frame, detection_result))
                else:
                    break

            cap.release()
            cv2.destroyAllWindows()

    def run(self, draw=False, display=False):
        t1 = threading.Thread(target=self.process_video, args=(draw, display))
        t1.start()

if __name__ == "__main__":
    personal_ai = PersonalAI()
    personal_ai.process_video(True, True)