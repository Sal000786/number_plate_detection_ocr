Number Plate Detection Project


This repository contains a Python script for detecting number plates in images. The project utilizes OpenCV for image processing tasks such as converting to grayscale, denoising, blurring, and edge detection. The script then applies the approxPolyDP method to detect the location of the number plate. Finally, a mask is created, and the masked image is passed to EasyOCR for number plate text detection.

Project Structure
easy_ocr_working.py and ANPR tutorial: The main Python script for number plate detection.
requirements.txt: List of dependencies required to run the project.
Prerequisites
Before running the script, ensure you have the required dependencies installed. You can install them using the following command:

pip install -r requirements.txt

Usage
Clone the repository to your local machine:
git clone https://github.com/yourusername/number-plate-detection.git
Navigate to the project directory:
cd easy_ocr_working
Run the number plate detection script:
python easy_ocr_working.py --image path/to/your/image.jpg
Replace path/to/your/image.jpg with the actual path to the image you want to process.

Number Plate Detection Process
The ANPR-tutorial.py script follows the following steps:

Image Preprocessing:

Converts the image to grayscale.
Applies denoising to reduce noise in the image.
Applies blurring for smoother edges.
Edge Detection:

Uses the Canny edge detector to highlight edges in the image.
Number Plate Location Detection:

Applies the approxPolyDP method to detect the location of the number plate based on edges.
Mask Creation:

Creates a mask for the detected number plate region.
EasyOCR for Number Plate Text Detection:

Passes the masked image containing the number plate to EasyOCR for text detection.
Displays the detected text on the console.

License
This project is licensed under the MIT License.
