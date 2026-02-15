import cv2, time

def save_snapshot(frame, label):
    name = f"snapshots/{label}_{int(time.time())}.jpg"
    cv2.imwrite(name, frame)

def log_event(msg):
    with open("logs.txt","a") as f:
        f.write(msg+"\n")
