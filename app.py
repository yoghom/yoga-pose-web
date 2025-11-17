# app.py
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

mp_pose = mp.solutions.pose

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def euclid(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def check_tadasana(landmarks):
    mp_l = mp_pose.PoseLandmark
    L_SH = landmarks[mp_l.LEFT_SHOULDER.value]
    R_SH = landmarks[mp_l.RIGHT_SHOULDER.value]
    L_EL = landmarks[mp_l.LEFT_ELBOW.value]
    R_EL = landmarks[mp_l.RIGHT_ELBOW.value]
    L_WR = landmarks[mp_l.LEFT_WRIST.value]
    R_WR = landmarks[mp_l.RIGHT_WRIST.value]
    L_HIP = landmarks[mp_l.LEFT_HIP.value]
    R_HIP = landmarks[mp_l.RIGHT_HIP.value]
    L_KNEE = landmarks[mp_l.LEFT_KNEE.value]
    R_KNEE = landmarks[mp_l.RIGHT_KNEE.value]
    L_ANK = landmarks[mp_l.LEFT_ANKLE.value]
    R_ANK = landmarks[mp_l.RIGHT_ANKLE.value]

    fb = []

    # Palms joined above head
    mid_hands_y = (L_WR.y + R_WR.y) / 2.0
    mid_sh_y = (L_SH.y + R_SH.y) / 2.0
    wrist_dist = euclid((L_WR.x, L_WR.y), (R_WR.x, R_WR.y))

    if not (mid_hands_y < mid_sh_y and wrist_dist < 0.08):
        fb.append("Raise hands and join palms above head")

    # Elbow angles
    L_ang = angle((L_SH.x, L_SH.y), (L_EL.x, L_EL.y), (L_WR.x, L_WR.y))
    R_ang = angle((R_SH.x, R_SH.y), (R_EL.x, R_EL.y), (R_WR.x, R_WR.y))
    if not (160 <= L_ang <= 180) or not (160 <= R_ang <= 180):
        fb.append("Keep elbows straight (no bending)")

    # Torso alignment
    mid_sh_x = (L_SH.x + R_SH.x) / 2.0
    mid_hp_x = (L_HIP.x + R_HIP.x) / 2.0
    if abs(mid_sh_x - mid_hp_x) > 0.035:
        fb.append("Keep your torso upright and centered")

    # Leg check
    thigh_r = ((R_KNEE.x + R_HIP.x)/2, (R_KNEE.y + R_HIP.y)/2)
    thigh_l = ((L_KNEE.x + L_HIP.x)/2, (L_KNEE.y + L_HIP.y)/2)

    d1 = euclid((L_ANK.x,L_ANK.y), thigh_r)
    d2 = euclid((R_ANK.x,R_ANK.y), thigh_l)

    if not ((d1 < 0.24 and L_ANK.y < R_KNEE.y - 0.02) or
            (d2 < 0.24 and R_ANK.y < L_KNEE.y - 0.02)):
        fb.append("Place one foot near inner thigh of standing leg")

    # Facing camera
    if abs(L_SH.x - R_SH.x) <= 0.02:
        fb.append("Face the camera directly")

    return fb if fb else ["Good Tadasana Posture!"]


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    with mp_pose.Pose() as pose:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            return {"feedback": ["Pose not detected — stand fully in frame."]}

        landmarks = res.pose_landmarks.landmark
        feedback = check_tadasana(landmarks)

        return {"feedback": feedback}


@app.get("/")
def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
