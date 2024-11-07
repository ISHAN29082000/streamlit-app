# Streamlit setup
import streamlit as st
from base64 import b64decode
import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

# Function for advanced color distribution using K-means clustering
def advanced_color_analysis(image):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    black_cluster = np.argmin(kmeans.cluster_centers_.sum(axis=1))
    black_pixels = np.sum(labels == black_cluster)
    white_pixels = len(labels) - black_pixels
    total_pixels = len(labels)
    black_percentage = (black_pixels / total_pixels) * 100
    white_percentage = (white_pixels / total_pixels) * 100
    return black_percentage, white_percentage

# Function to assign grades based on color distribution
def assign_grade(black_percentage, white_percentage):
    if black_percentage > white_percentage:
        return 'Grade C: More coke (black)'
    elif white_percentage > black_percentage:
        return 'Grade A: More cryolite (white)'
    else:
        return 'Grade B: Balanced'

# Main function
def main():
    st.title("Coke-Cryolite Image Analysis")

    # Upload image through Streamlit interface
    uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Load image with OpenCV and resize for faster processing
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = cv2.resize(image, (512, 512))

        # Display the image
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Analyze the color distribution using advanced K-means clustering
        black_percentage, white_percentage = advanced_color_analysis(image)

        # Assign a grade based on the distribution
        grade = assign_grade(black_percentage, white_percentage)

        # Display the results
        st.write(f'**Black (Coke) Percentage:** {black_percentage:.2f}%')
        st.write(f'**White (Cryolite) Percentage:** {white_percentage:.2f}%')
        st.write(f'**Assigned Grade:** {grade}')

# Run the app
if __name__ == "__main__":
    main()
