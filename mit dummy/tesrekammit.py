import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import numpy as np
import time
import math

# =========================
# FIREBASE
# =========================

cred = credentials.Certificate("firebase-key.json")

firebase_admin.initialize_app(
    cred,
    {
        "databaseURL":
        "https://smart-ekg-tmj-6b-default-rtdb.asia-southeast1.firebasedatabase.app"
    }
)

print("Firebase Connected")

# =========================
# DUMMY ECG LOOP
# =========================

t = 0

while True:

    ecg = []

    for i in range(50):

        x = (t + i) % 100

        # baseline
        value = 700 + 10 * math.sin(x * 0.2)

        # spike QRS
        if x == 20:
            value = 1000
        elif x == 19 or x == 21:
            value = 850
        elif x == 18 or x == 22:
            value = 750

        ecg.append(int(value))

    payload = {
        "bpm": 80,
        "graph_value": ecg,
        "status": "online",
        "timestamp": int(time.time())
    }

    db.reference(
        "monitoring/live_data"
    ).set(payload)

    print("Sent BPM 80")

    t += 5

    time.sleep(0.2)