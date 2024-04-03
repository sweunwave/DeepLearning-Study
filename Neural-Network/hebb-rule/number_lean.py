import cv2
import numpy as np

def main():
    image_1 = np.array([[0, 1],
                        [1, 0]])

    cv2.imshow('bit_image', image_1)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()