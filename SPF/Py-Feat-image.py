from feat.utils.io import get_test_data_path
from feat.plotting import imshow
import os
from pathlib import Path
from feat import Detector

detector = Detector()


# Helper to point to the test data folder
test_data_dir = Path(__file__).parent

# Get the full path
single_face_img_path = os.path.join(test_data_dir, "single_face_sad.png")

# Plot it
imshow(single_face_img_path)

single_face_prediction = detector.detect_image(single_face_img_path, data_type="image")

# Check the type
print(type(single_face_prediction)) 

# Show results
print(single_face_prediction.emotions)
print(single_face_prediction.aus)

single_face_prediction.to_csv("image_output.csv", index=False)