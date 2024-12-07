from flask import Flask, request, jsonify
from nudenet import NudeDetector
import cv2
import os
import json
import joblib

# Tạo Flask app
app = Flask(__name__)

# Load mô hình toxic comment
try:
    toxic_model = joblib.load('toxic_comment_model.pkl')
except FileNotFoundError:
    raise FileNotFoundError("Không tìm thấy file mô hình 'toxic_comment_model.pkl'.")

# Nhãn cho toxic comment
LABELS = ["Độc hại", "Rất độc hại", "Thô tục", "Đe dọa", "Xúc phạm", "Kỳ thị"]

# Khởi tạo NudeDetector
detector = NudeDetector()

# Các lớp nội dung nhạy cảm
sensitive_classes = [
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED"
]
threshold = 0.5

# Đảm bảo JSON trả về với mã hóa UTF-8
@app.after_request
def after_request(response):
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return response

# Endpoint dự đoán toxic comment
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Nhận dữ liệu từ yêu cầu
        data = request.json
        comment = data.get('comment', '')

        if not isinstance(comment, str) or not comment.strip():
            return jsonify({'error': 'Bình luận hợp lệ là bắt buộc.'}), 400

        # Dự đoán bằng mô hình
        prediction = toxic_model.predict([comment])
        prediction = prediction[0]

        # Lấy nhãn cho các dự đoán dương tính
        result_labels = [LABELS[i] for i, value in enumerate(prediction) if value == 1]
        if not result_labels:
            result_labels = ["Bình thường"]

        # Tạo phản hồi
        response = {
            'comment': comment,
            'predicted_labels': result_labels
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': f"Đã xảy ra lỗi: {str(e)}"}), 500

# Endpoint kiểm tra ảnh
@app.route('/check_image', methods=['POST'])
def check_image():
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "Không có tệp ảnh được cung cấp."}), 400

    # Lưu và xử lý ảnh
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Phát hiện nội dung nhạy cảm
    result = detector.detect(file_path)
    is_sensitive = any(
        item["class"] in sensitive_classes and item["score"] >= threshold
        for item in result
    )

    # Xóa tệp sau khi xử lý
    os.remove(file_path)

    # Trả về kết quả
    response = {"message": "Ảnh khiêu dâm."} if is_sensitive else {"message": "Ảnh bình thường."}
    return jsonify(response)

# Endpoint kiểm tra video
@app.route('/check_video', methods=['POST'])
def check_video():
    file = request.files.get('video')
    if not file:
        return jsonify({"error": "Không có tệp video được cung cấp."}), 400

    # Lưu video
    file_path = os.path.join('uploads', file.filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(file_path)

    # Xử lý video
    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = 5  # Kiểm tra mỗi 5 khung hình
    frame_count = 0
    is_sensitive = False
    resize_width, resize_height = 320, 240
    message = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (resize_width, resize_height))

        result = detector.detect(frame_resized)
        is_sensitive = any(
            item["class"] in sensitive_classes and item["score"] >= threshold
            for item in result
        )

        if is_sensitive:
            seconds = frame_count / fps
            message = f"Phát hiện khung hình nhạy cảm tại giây {seconds:.2f}."
            break

    cap.release()
    os.remove(file_path)

    # Trả về kết quả
    response = {
        "message": "Nội dung khiêu dâm." if is_sensitive else "Nội dung bình thường.",
        "details": message if is_sensitive else None
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
