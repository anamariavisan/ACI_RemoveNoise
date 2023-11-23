from random import random, seed
from sys import stdin
import cv2
import numpy as np

def add_salt_and_pepper(image, proc, add_pepper):
    grayscale_img = image

    if len(image.shape) == 3:
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    width, height = grayscale_img.shape

    for i in range(width):
        for j in range(height):
            p = random()

            if p < proc:
                if p < proc / 2:
                    grayscale_img[i][j] = 255
                else:
                    if add_pepper:
                        grayscale_img[i][j] = 0
                    

    cv2.imshow("image", image)
    cv2.imshow("noise image", grayscale_img)
    cv2.waitKey(0)
    return grayscale_img

def add_gaussian(image, mean, std):
    grayscale_img = image

    if len(image.shape) == 3:
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    width, height = grayscale_img.shape
    gaussian = np.random.normal(mean, std, (width, height))

    grayscale_img = grayscale_img + gaussian

    cv2.normalize(grayscale_img, grayscale_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    grayscale_img = grayscale_img.astype(np.uint8)

    cv2.imshow("image", image)
    cv2.imshow("noise image", grayscale_img)
    cv2.waitKey(0)
    return grayscale_img

def add_speckle(image):
    grayscale_img = image

    if len(image.shape) == 3:
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    width, height = grayscale_img.shape
    noise = np.random.randn(width, height)

    grayscale_img = grayscale_img + noise * grayscale_img

    cv2.normalize(grayscale_img, grayscale_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    grayscale_img = grayscale_img.astype(np.uint8)

    cv2.imshow("image", image)
    cv2.imshow("noise image", grayscale_img)
    cv2.waitKey(0)
    return grayscale_img

def add_poisson(image, lam):
    grayscale_img = image

    if len(image.shape) == 3:
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    width, height = grayscale_img.shape
    noise = np.random.poisson(lam, (width, height))

    grayscale_img = grayscale_img + noise 

    cv2.normalize(grayscale_img, grayscale_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    grayscale_img = grayscale_img.astype(np.uint8)

    cv2.imshow("image", image)
    cv2.imshow("noise image", grayscale_img)
    cv2.waitKey(0)
    return grayscale_img

def add_rayleigh(image, scale):
    grayscale_img = image

    if len(image.shape) == 3:
        grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    width, height = grayscale_img.shape
    noise = np.random.rayleigh(scale, (width, height))

    grayscale_img = grayscale_img + noise 

    cv2.normalize(grayscale_img, grayscale_img, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    grayscale_img = grayscale_img.astype(np.uint8)

    cv2.imshow("image", image)
    cv2.imshow("noise image", grayscale_img)
    cv2.waitKey(0)
    return grayscale_img

if __name__ == '__main__':
    seed(5)
    file = None
    file_exists = False

    while True:
        print("Enter command: (READ/ADD/REMOVE/EXIT)")
        cmd = stdin.readline().rstrip()

        if cmd == "READ":
            print("Enter filename:")
            filename = stdin.readline().rstrip()
            file = cv2.imread(filename, cv2.IMREAD_COLOR)
            cv2.imshow("image", file)
            cv2.waitKey(0)
            file_exists = True
            continue

        if cmd == "ADD":
            if not file_exists:
                print("Enter filename:")
                filename = stdin.readline().rstrip()
                file = cv2.imread(filename, cv2.IMREAD_COLOR)
                cv2.imshow("image", file)
                cv2.waitKey(0)
                file_exists = True

            print("Enter noise type:")
            noise_type = stdin.readline().rstrip()

            if noise_type == "SALT_PEPPER":
                file = add_salt_and_pepper(file, 0.1, True)
                continue

            if noise_type == "GAUSSIAN":
                file = add_gaussian(file, 2, 5)
                continue

            if noise_type == "UNIFORM":
                file = add_salt_and_pepper(file, 0.1, False)
                continue

            if noise_type == "SPECKLE":
                file = add_speckle(file)
                continue

            if noise_type == "POISSON":
                file = add_poisson(file, 64)
                continue

            if noise_type == "RAYLEIGH":
                file = add_rayleigh(file, 64)
                continue

            print ("noise type undefined")
            continue

        if cmd == "REMOVE":
            if not file_exists:
                print("Enter filename:")
                filename = stdin.readline().rstrip()
                file = cv2.imread(filename, cv2.IMREAD_COLOR)
                cv2.imshow("image", file)
                cv2.waitKey(0)
                file_exists = True

            print("Enter method:")
            method = stdin.readline().rstrip()

            if method == "GAUSSIAN":
                image = cv2.GaussianBlur(file, (5, 5), cv2.BORDER_DEFAULT)
                cv2.imshow("image", file)
                cv2.imshow("Gausian image", image)
                cv2.waitKey(0)
                continue

            if method == "MEDIAN":
                image = cv2.medianBlur(file, 5)
                cv2.imshow("image", file)
                cv2.imshow("Gausian image", image)
                cv2.waitKey(0)
                continue

            if method == "BILATERAL":
                image = cv2.bilateralFilter(file, 9, 75, 75)
                cv2.imshow("image", file)
                cv2.imshow("Gausian image", image)
                cv2.waitKey(0)
                continue

            if method == "NONLOCAL_MEANS":
                image = cv2.fastNlMeansDenoising(file, None, 15, 7, 21)
                cv2.imshow("image", file)
                cv2.imshow("Gausian image", image)
                cv2.waitKey(0)
                continue

            print("method undefined")
            continue

        if cmd == "EXIT":
            break

        print("command undefined")
