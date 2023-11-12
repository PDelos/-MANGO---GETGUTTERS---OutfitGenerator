import cv2
import numpy as np
import random

def get_dominant_colors(image, num_colors):
    # Reshape the image to be a list of pixels
    pixels = image.reshape((-1, 3))

    # Apply k-means clustering to find dominant colors
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the RGB values of dominant colors to integers
    dominant_colors = np.uint8(centers)

    return dominant_colors

def generate_complementary_color(base_color):
    # Calculate the complementary color
    complementary_color = cv2.subtract(np.array([255, 255, 255], dtype=np.uint8), base_color)
    return complementary_color

def generate_monochromatic_colors(base_color, num_colors):
    hsv_base = cv2.cvtColor(np.uint8([[base_color]]), cv2.COLOR_BGR2HSV)[0][0]
    monochromatic_colors = []

    for _ in range(num_colors):
        # Generate random brightness values
        random_value = random.randint(0, 255)

        # Create a monochromatic color in HSV space
        monochromatic_hsv_color = np.array([hsv_base[0], hsv_base[1], random_value], dtype=np.uint8)

        # Convert back to BGR color space
        monochromatic_bgr_color = cv2.cvtColor(np.uint8([[monochromatic_hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]

        monochromatic_colors.append(monochromatic_bgr_color)

    return monochromatic_colors

def generate_analogous_colors(base_color, num_colors):
    # Calculate analogous colors by varying hue
    hsv_base = cv2.cvtColor(np.uint8([[base_color]]), cv2.COLOR_BGR2HSV)[0][0]
    analogous_colors = []

    for _ in range(num_colors):
        # Generate random hue values within a range
        random_hue = random.randint(hsv_base[0] - 30, hsv_base[0] + 30) % 180

        # Create an analogous color in HSV space
        analogous_hsv_color = np.array([random_hue, hsv_base[1], hsv_base[2]], dtype=np.uint8)

        # Convert back to BGR color space
        analogous_bgr_color = cv2.cvtColor(np.uint8([[analogous_hsv_color]]), cv2.COLOR_HSV2BGR)[0][0]

        analogous_colors.append(analogous_bgr_color)

    return analogous_colors

def save_concatenated_image(image, file_path):
    cv2.imwrite(file_path, image)

def open_image_with_viewer(file_path):
    import subprocess
    try:
        subprocess.run(["xdg-open", file_path])  # Use "open" on macOS
    except Exception as e:
        print(f"Error opening image with viewer: {e}")


def main(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Get dominant colors from the image (you can adjust the number of colors)
    num_dominant_colors = int(input("Enter the number of dominant colors to show: "))
    dominant_colors = get_dominant_colors(image, num_dominant_colors)

    # Create a canvas for displaying the original image
    canvas_original = np.copy(image)

    # Generate and display random colors with BGR values
    num_random_colors = int(input("Enter the number of random colors to show: "))

    # Create a canvas for displaying random colors and BGR values with a white background
    canvas_random_colors = np.ones((image.shape[0], 350, 3), dtype=np.uint8) * 255

    # Display dominant colors and their BGR values
    for i, color in enumerate(dominant_colors):
        color_rect = np.ones((40, 40, 3), dtype=np.uint8) * color
        cv2.putText(canvas_random_colors, f"Dominant Colour {i+1}:", (25, 20 + i * 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(canvas_random_colors, (60, 40 + i * 70), (100, 80 + i * 70), tuple(map(int, color)), -1)
        cv2.putText(canvas_random_colors, f"BGR {tuple(color)}", (120, 70 + i * 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Generate and display random colors with BGR values
    for i in range(num_random_colors):
        base_color = dominant_colors[i % num_dominant_colors]

        # Generate complementary, monochromatic, or analogous color
        random_choice = random.choice(["complementary", "monochromatic", "analogous"])
        if random_choice == "complementary":
            random_color = generate_complementary_color(base_color)
            color_name = "Complementary"
        elif random_choice == "monochromatic":
            random_color = generate_monochromatic_colors(base_color, 1)[0]
            color_name = "Monochromatic"
        else:
            random_color = generate_analogous_colors(base_color, 1)[0]
            color_name = "Analogous"

        cv2.putText(canvas_random_colors, f"{color_name} Color {i+1}:", (20, 25 + (num_dominant_colors + i) * 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(canvas_random_colors, (60, 40 + (num_dominant_colors + i) * 70),
                      (100, 80 + (num_dominant_colors + i) * 70), tuple(map(int, random_color)), -1)
        cv2.putText(canvas_random_colors, f"BGR {tuple(random_color)}", (120, 70 + (num_dominant_colors + i) * 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # Concatenate the original image and the canvas with random colors
    concatenated_image = np.hstack((canvas_original, canvas_random_colors))

    # Save the concatenated image to a file
    output_image_path = "concatenated_image.png"
    save_concatenated_image(concatenated_image, output_image_path)

    # Open the saved image with an external viewer
    open_image_with_viewer(output_image_path)

if __name__ == "__main__":
    image_path = '../dataset/outfits/1.png' # Update your file path
    main(image_path)
