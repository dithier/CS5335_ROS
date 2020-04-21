import numpy as np
import cv2

def main():
    with np.load("all_images.npz") as data:
        img_array = data["images"]

    n = len(img_array)
    # img_array = np.array(img_array)

    np.random.seed(0)
    indices = np.random.randint(0, n - 1, size=50)
    selected_imgs = img_array[indices]

    print(len(selected_imgs))
    np.savez("50_images", images=selected_imgs)

if __name__ == "__main__":
    main()