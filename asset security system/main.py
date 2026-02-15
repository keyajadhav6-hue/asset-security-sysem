import cv2
import time
from datetime import datetime
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------------- MODEL & TRACKER ----------------
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)
class_names = model.names

cap = cv2.VideoCapture(0)

# ---------------- VIDEO RECORD ----------------
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(
    f"recording_{int(time.time())}.avi",
    fourcc, 20.0,
    (int(cap.get(3)), int(cap.get(4)))
)

# ---------------- CONFIG ----------------
ABANDON_TIME = 10
REMOVAL_TIME = 15

object_info = {}   # {id: {"pos":(x,y), "time":t, "cls":cls}}
appeared_ids = set()
abandoned_ids = set()
removed_ids = set()

# ---------------- FUNCTIONS ----------------
def log_event(msg):
    with open("logs.txt", "a") as f:
        f.write(msg + "\n")

def timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

    for box in results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        detections.append(([x1,y1,x2-x1,y2-y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)
    current_ids = set()

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l,t,r,b = map(int, track.to_ltrb())
        cx = (l+r)//2
        cy = (t+b)//2
        obj_cls = track.det_class if track.det_class is not None else 0
        obj_name = class_names[obj_cls]

        current_ids.add(track_id)

        # First time appeared
        if track_id not in object_info:
            object_info[track_id] = {
                "pos": (cx,cy),
                "time": time.time(),
                "cls": obj_cls
            }

            if track_id not in appeared_ids:
                log_event(f"[{timestamp()}] Object appeared: {obj_name} (ID {track_id})")
                appeared_ids.add(track_id)

        else:
            prev_pos = object_info[track_id]["pos"]

            # Check abandoned
            if abs(prev_pos[0]-cx) < 10 and abs(prev_pos[1]-cy) < 10:
                if time.time() - object_info[track_id]["time"] > ABANDON_TIME:
                    if track_id not in abandoned_ids:
                        log_event(f"[{timestamp()}] Abandoned object: {obj_name} (ID {track_id})")
                        abandoned_ids.add(track_id)
            else:
                object_info[track_id]["pos"] = (cx,cy)
                object_info[track_id]["time"] = time.time()

        cv2.rectangle(frame, (l,t), (r,b), (0,255,0), 2)
        cv2.putText(frame, f"{obj_name} ID {track_id}",
                    (l, t-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0), 2)

    # Removal detection
    for oid in list(object_info.keys()):
        if oid not in current_ids:
            if time.time() - object_info[oid]["time"] > REMOVAL_TIME:
                if oid not in removed_ids:
                    obj_name = class_names[object_info[oid]["cls"]]
                    log_event(f"[{timestamp()}] Object removed: {obj_name} (ID {oid})")
                    removed_ids.add(oid)
                del object_info[oid]

    # Show time on camera
    cv2.putText(frame, timestamp(), (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0,0,255), 2)

    cv2.imshow("Object & Asset Security Monitoring", frame)
    out.write(frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
