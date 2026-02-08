import cv2
from flask import Flask, render_template, Response, jsonify, send_file
import io
import numpy as np
import onnxruntime as ort
from gtts import gTTS
from tamil_map import TAMIL_MAP
from llm import guess_tamil_word

app = Flask(__name__)

# Load ONNX model
session = ort.InferenceSession("best.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Get input shape (typically [1, 3, 640, 640])
input_shape = session.get_inputs()[0].shape
img_height, img_width = int(input_shape[2]), int(input_shape[3])

camera_running = False
current_letter = ""
final_letters = []
sentence_words = []

def preprocess_image(img, input_size):
    """Preprocess image for YOLO ONNX model"""
    # Resize with letterbox (maintain aspect ratio)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = letterbox(img_rgb, input_size)[0]
    
    # Normalize to [0, 1] and convert to float32
    img_resized = img_resized.astype(np.float32) / 255.0
    
    # Transpose HWC to CHW
    img_transposed = np.transpose(img_resized, (2, 0, 1))
    
    # Add batch dimension
    img_batch = np.expand_dims(img_transposed, axis=0)
    
    return img_batch

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resize image with unchanged aspect ratio using padding"""
    shape = img.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)

def gen_frames():
    global camera_running, current_letter, final_letters

    cap = cv2.VideoCapture(0)

    while camera_running:
        success, frame = cap.read()
        if not success:
            break

        original_img = frame.copy()
        img_height_orig, img_width_orig = original_img.shape[:2]
        
        # Preprocess
        input_tensor = preprocess_image(original_img, (img_height, img_width))
        
        # Run inference
        outputs = session.run(None, {input_name: input_tensor})[0]
        
        # Post-process
        detected = ""
        boxes = postprocess(outputs, img_width_orig, img_height_orig, img_width, img_height, conf_threshold=0.6)
        
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box
            cls_id = int(cls_id) + 1
            detected = TAMIL_MAP.get(cls_id, "")
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{detected} {conf:.2f}", (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # ---- collapse repeated letters ----
        if detected:
            if not final_letters or detected != final_letters[-1]:
                final_letters.append(detected)
            current_letter = detected

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()

def postprocess(output, img_width, img_height, input_width, input_height, conf_threshold=0.6):
    """Post-process YOLO ONNX output"""
    # Output shape: (1, 252, 8400) -> (8400, 252)
    predictions = np.squeeze(output).T
    
    # Get scaling factor
    scale = min(input_width / img_width, input_height / img_height)
    
    boxes = []
    for pred in predictions:
        # Extract box and class scores
        x_center, y_center, w, h = pred[:4]
        class_scores = pred[4:]
        
        # Get class with highest score
        cls_id = np.argmax(class_scores)
        confidence = class_scores[cls_id]
        
        if confidence >= conf_threshold:
            # Convert from center format to corner format and scale to original image
            x1 = (x_center - w / 2) / scale
            y1 = (y_center - h / 2) / scale
            x2 = (x_center + w / 2) / scale
            y2 = (y_center + h / 2) / scale
            
            # Clip to image boundaries
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
            
            boxes.append([x1, y1, x2, y2, confidence, cls_id])
    
    return boxes

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
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )


