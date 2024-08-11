# app.py
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)

def load_and_preprocess_image(image_file, blur_kernel_size=(5, 5)):
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_kernel_size, 0)
    return image, gray, blurred

def detect_edges(image, low_threshold=50, high_threshold=150):
    return cv2.Canny(image, low_threshold, high_threshold)

def find_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def smooth_contour(contour, window_size=5):
    smoothed_contour = np.copy(contour)
    for i in range(len(contour)):
        start = max(0, i - window_size // 2)
        end = min(len(contour), i + window_size // 2 + 1)
        smoothed_contour[i] = np.mean(contour[start:end], axis=0)
    return smoothed_contour

def check_reflection_symmetry(contour, axis='vertical', tolerance=0.05):
    h, w = np.max(contour, axis=0)[0] - np.min(contour, axis=0)[0]
    centroid = np.mean(contour[:, 0, :], axis=0)

    if axis == 'vertical':
        contour_mirror = contour.copy()
        contour_mirror[:, 0, 0] = 2 * centroid[0] - contour[:, 0, 0]
    elif axis == 'horizontal':
        contour_mirror = contour.copy()
        contour_mirror[:, 0, 1] = 2 * centroid[1] - contour[:, 0, 1]
    
    distances = np.linalg.norm(contour[:, 0, :] - contour_mirror[:, 0, :], axis=1)
    symmetry_score = np.mean(distances)
    
    return symmetry_score < tolerance, symmetry_score

def analyze_shapes(image_file):
    image, gray, blurred = load_and_preprocess_image(image_file)
    edges = detect_edges(blurred)
    contours = find_contours(edges)
    
    shape_img = image.copy()
    results = []
    
    for contour in contours:
        if len(contour) < 5:
            continue
        
        contour = smooth_contour(contour)
        
        x, y, w, h = cv2.boundingRect(contour)
        extracted_shape = gray[y:y+h, x:x+w]
        
        symmetric_vertical, score_vertical = check_reflection_symmetry(contour, axis='vertical')
        symmetric_horizontal, score_horizontal = check_reflection_symmetry(contour, axis='horizontal')
        
        symmetry_status = "Symmetric"
        if not symmetric_vertical and not symmetric_horizontal:
            symmetry_status = "Not Symmetric"
        elif symmetric_vertical:
            symmetry_status += " (Vertical)"
        elif symmetric_horizontal:
            symmetry_status += " (Horizontal)"
        
        shape_name = "Closed Shape"
        
        cv2.drawContours(shape_img, [contour], -1, (0, 255, 0), 2)
        cv2.putText(shape_img, f"{shape_name}, {symmetry_status}", (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        results.append({
            'shape': shape_name,
            'symmetric': symmetry_status,
            'score_vertical': float(score_vertical),
            'score_horizontal': float(score_horizontal),
        })
    
    # Convert the analyzed image to base64
    _, buffer = cv2.imencode('.png', shape_img)
    analyzed_image = base64.b64encode(buffer).decode('utf-8')
    
    return analyzed_image, results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        analyzed_image, results = analyze_shapes(image_file)
        return jsonify({
            'analyzed_image': analyzed_image,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)