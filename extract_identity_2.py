import face_recognition
import cv2
import numpy as np

# Step 1: Load image from webcam
def capture_face_image():
    cam = cv2.VideoCapture(0)
    print("[INFO] Starting camera. Press SPACE to capture.")
    while True:
        ret, frame = cam.read()
        cv2.imshow("Press SPACE to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):  # Press Space to capture
            img = frame
            break
    cam.release()
    cv2.destroyAllWindows()
    return img

# Step 2: Extract face embedding
def extract_face_vector(image):
    face_locations = face_recognition.face_locations(image)
    if len(face_locations) == 0:
        raise ValueError("No face detected.")
    embeddings = face_recognition.face_encodings(image, face_locations)
    return embeddings[0]  # Return the first face

# Step 3: Normalize the vector
def normalize_vector(vector):
    return vector / np.linalg.norm(vector)

# Step 4: Save vector
def save_vector(vector, filename="identity_vector.npy"):
    np.save(filename, vector)
    print(f"[INFO] Identity vector saved to {filename}")

# --- Main Logic ---
if __name__ == "__main__":
    try:
        face_img = capture_face_image()
        raw_vector = extract_face_vector(face_img)
        normalized_vector = normalize_vector(raw_vector)

        print("[INFO] Raw Vector Sample:", raw_vector[:5])
        print("[INFO] Normalized Vector Sample:", normalized_vector[:5])
        print("[INFO] Norm:", np.linalg.norm(normalized_vector))

        save_vector(normalized_vector)

    except Exception as e:
        print("[ERROR]", e)
save_vector(normalized_vector, filename="identity_vector_2.npy")
