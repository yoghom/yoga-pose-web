# app.py
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from cvzone.PoseModule import PoseDetector

detector = PoseDetector(detectionCon=0.7, trackCon=0.6)

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


def check_tadasana(lm):
    # cvzone returns list of 33 landmarks: each = (x, y, z)
    L_SH = lm[11]
    R_SH = lm[12]
    L_EL = lm[13]
    R_EL = lm[14]
    L_WR = lm[15]
    R_WR = lm[16]
    L_HIP = lm[23]
    R_HIP = lm[24]
    L_KNEE = lm[25]
    R_KNEE = lm[26]
    L_ANK = lm[27]
    R_ANK = lm[28]

    fb = []

    # Palms joined above head
    mid_hands_y = (L_WR[1] + R_WR[1]) / 2
    mid_sh_y = (L_SH[1] + R_SH[1]) / 2
    wrist_dist = euclid((L_WR[0], L_WR[1]), (R_WR[0], R_WR[1]))

    if not (mid_hands_y < mid_sh_y and wrist_dist < 40):
        fb.append("Raise hands and join palms above head")

    # Elbow angles
    L_ang = angle(L_SH[:2], L_EL[:2], L_WR[:2])
    R_ang = angle(R_SH[:2], R_EL[:2], R_WR[:2])
    if not (160 <= L_ang <= 180) or not (160 <= R_ang <= 180):
        fb.append("Keep elbows straight (no bending)")

    # Torso alignment
    mid_sh_x = (L_SH[0] + R_SH[0]) / 2
    mid_hp_x = (L_HIP[0] + R_HIP[0]) / 2
    if abs(mid_sh_x - mid_hp_x) > 20:
        fb.append("Keep your torso upright and centered")

    # Leg check
    thigh_r = ((R_KNEE[0] + R_HIP[0]) / 2, (R_KNEE[1] + R_HIP[1]) / 2)
    thigh_l = ((L_KNEE[0] + L_HIP[0]) / 2, (L_KNEE[1] + L_HIP[1]) / 2)

    d1 = euclid((L_ANK[0], L_ANK[1]), thigh_r)
    d2 = euclid((R_ANK[0], R_ANK[1]), thigh_l)

    if not ((d1 < 120 and L_ANK[1] < R_KNEE[1] - 15) or
            (d2 < 120 and R_ANK[1] < L_KNEE[1] - 15)):
        fb.append("Place one foot near inner thigh of standing leg")

    # Facing camera
    if abs(L_SH[0] - R_SH[0]) <= 10:
        fb.append("Face the camera directly")

    return fb if fb else ["Good Tadasana Posture!"]


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    img = detector.findPose(frame, draw=False)
    lmList, _ = detector.findPosition(img, draw=False)

    if not lmList or len(lmList) < 29:
        return {"feedback": ["Pose not detected — stand fully in frame."]}

    feedback = check_tadasana(lmList)
    return {"feedback": feedback}


@app.get("/")
def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
