import numpy as np
import cv2


def main():
    # Create a black image
    img = np.zeros((1000, 1000, 3), np.uint8)

    # top-left corner, bottom-right corner
    cv2.rectangle(img, (335, 300), (665, 700), (255, 255, 255), -1)

    """
    rows, cols, channel = img.shape

    center_x = (335 + 665) / 2
    center_y = (300 + 700) / 2

    M = cv2.getRotationMatrix2D((center_x, center_y), 315, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('rotated image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    cv2.imwrite("../images/desired_cnt.png", img)

    # Create a black image
    img = np.zeros((1000, 1000, 3), np.uint8)

    # top-left corner, bottom-right corner
    cv2.rectangle(img, (335, 300), (664, 629), (255, 255, 255), -1)

    cv2.imwrite("../images/wrong_cnt.png", img)


if __name__ == "__main__":
    main()