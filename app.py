import cv2
from flask import Flask, render_template, Response, jsonify, send_file
import io
from ultralytics import YOLO
from gtts import gTTS
from tamil_map import TAMIL_MAP
from llm import guess_tamil_word

app = Flask(__name__)

model = YOLO("best.pt")

camera_running = False
current_letter = ""
final_letters = []
sentence_words = []

def gen_frames():
    global camera_running, current_letter, final_letters

    cap = cv2.VideoCapture(0)

    while camera_running:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.6)
        detected = ""

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0]) + 1
                detected = TAMIL_MAP.get(cls_id, "")

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # ---- collapse repeated letters ----
        if detected:
            if not final_letters or detected != final_letters[-1]:
                final_letters.append(detected)
            current_letter = detected

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_camera", methods=["POST"])
def start_camera():
    global camera_running, final_letters, current_letter
    camera_running = True
    final_letters = []
    current_letter = ""
    return jsonify({"status": "started"})

@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global camera_running
    camera_running = False
    return jsonify({"status": "stopped"})

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify({
        "previous": "".join(final_letters),
        "current": current_letter,
        "sentence": " ".join(sentence_words)
    })

@app.route("/finalize_word", methods=["POST"])
def finalize_word():
    global final_letters, sentence_words

    if final_letters:
        word = guess_tamil_word(final_letters)
        sentence_words.append(word)
        final_letters = []
        return jsonify({"sentence": " ".join(sentence_words)})

    return jsonify({"sentence": " ".join(sentence_words)})

@app.route("/speak", methods=["POST"])
def speak():
    text = " ".join(sentence_words)
    if not text:
        return "", 204

    tts = gTTS(text=text, lang="ta")
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)

    return send_file(
        mp3_fp,
        mimetype="audio/mpeg",
        as_attachment=False
    )

if __name__ == "__main__":
    app.run(debug=True)
