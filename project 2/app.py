from flask import Flask, request, render_template, send_file, redirect, url_for
import numpy as np
import cv2
import os
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def overlay_qr_on_image(host_image_path, qr_image_path):
    # Read the host image
    host_image = cv2.imread(host_image_path)
    if host_image is None:
        raise ValueError(f"Failed to load host image from {host_image_path}")

    # Read the QR code image
    qr_image = cv2.imread(qr_image_path, cv2.IMREAD_GRAYSCALE)
    if qr_image is None:
        raise ValueError(f"Failed to load QR image from {qr_image_path}")

    # Ensure the QR code is binary
    _, qr_binary = cv2.threshold(qr_image, 127, 255, cv2.THRESH_BINARY)

    # Resize the QR code image to a smaller size relative to the host image
    qr_size = min(host_image.shape[0], host_image.shape[1]) // 4  # Resize to 1/4th of the smallest dimension
    qr_image_resized = cv2.resize(qr_binary, (qr_size, qr_size))

    # Convert QR image to color
    qr_image_resized_colored = cv2.cvtColor(qr_image_resized, cv2.COLOR_GRAY2BGR)

    # Define the position where QR code will be placed (bottom-right corner)
    x_offset = host_image.shape[1] - qr_image_resized_colored.shape[1] - 10  # 10 pixels margin from the edges
    y_offset = host_image.shape[0] - qr_image_resized_colored.shape[0] - 10  # 10 pixels margin from the edges

    # Create two copies of the host image
    overlay = host_image.copy()
    output = host_image.copy()

    # Draw the QR code image onto the overlay
    overlay[y_offset:y_offset + qr_image_resized_colored.shape[0],
            x_offset:x_offset + qr_image_resized_colored.shape[1]] = qr_image_resized_colored

    # Apply the transparent overlay using cv2.addWeighted
    alpha = 0.5  # adjust the opacity value (0.0 to 1.0)
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # Save the combined image
    combined_image_path = 'static/combined_image.png'
    cv2.imwrite(combined_image_path, output)

    return combined_image_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        host_image = request.files['host_image']
        qr_image = request.files['qr_image']

        if not host_image or not qr_image:
            return "Please upload both the host image and QR image."

        host_image_path = os.path.join('static', 'host_image.png')
        qr_image_path = os.path.join('static', 'qr_image.png')

        host_image.save(host_image_path)
        qr_image.save(qr_image_path)

        try:
            combined_image_path = overlay_qr_on_image(host_image_path, qr_image_path)
            return send_file(combined_image_path, mimetype='image/png')
        except Exception as e:
            logging.error("An error occurred: %s", e)
            return str(e), 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)