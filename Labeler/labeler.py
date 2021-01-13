import cv2
from os import listdir
from os.path import isfile, join
import os
import getch
import numpy as np

def GetRectangle(image):
    ROI = cv2.selectROI("select ROI", image)
    pt1 = (ROI[0], ROI[1])
    pt2 = (ROI[0] + ROI[2], ROI[1] + ROI[3])
    rect = (pt1[0], pt1[1], pt2[0], pt2[1])
    return rect

def GetPoints(image):
    ROI = cv2.selectROI("select ROI", image)
    pt1 = (ROI[0], ROI[1])
    pt2 = (ROI[0] + ROI[2], ROI[1] + ROI[3])
    return pt1, pt2

def ResizeBoundingBox(coords):
    return tuple(2 * coord for coord in coords)

# Get Dirs to images
IMAGES_DIR = r"./Images"
OUTPUT_DIR_LABELS = r"./Segmented"
OUTPUT_DIR_IMAGES = r"./DONE_PNG"
IMAGES_DONE_DIR = r"./DONE_INPUTS"
# Get all paths
inputPaths = [join(IMAGES_DIR, f) for f in listdir(IMAGES_DIR) if isfile(join(IMAGES_DIR, f))]
# Make break image
BREAK_IMAGE = "That's done!"
breakImg = np.ones((224, 224, 3))
cv2.putText(breakImg, "THAT ONE'S DONE ;-)", (25, 120), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0, 255))
cv2.putText(breakImg, "1 to Update", (50, 140), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0, 255))
cv2.putText(breakImg, "Any to Confirm", (50, 160), cv2.FONT_HERSHEY_COMPLEX, .5, (0, 0, 255))

# Constants
IMAGE_INPUT_WINDOW = "Input"
IMAGE_THRESH_WINDOW = "Threshold test"
CURRENT_OBJECT_WINDOW = "Current object"
LABELED_WINDOW = "Labeled Image"
MULTIPLIED_WINDOW = "Multiplied Image"
RESIZED_IMAGE_WINDOW = "Base Image"

for i in range(len(inputPaths)):
    image = cv2.imread(inputPaths[i])
    print(inputPaths[i])
    imageRes = cv2.resize(image, (0, 0), fx=2, fy=2)

    threshold = 120

    print("+ to increase by 1\n- to decrease by 1\n] to increase by 10\n[ to decrease by 10\n ENTER to accept")
    imageLabeled = np.zeros(image.shape)
    imageLabeledMulti = np.zeros(image.shape)
    while True:
        while True:
            _, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)

            cv2.imshow(RESIZED_IMAGE_WINDOW, imageRes)
            cv2.moveWindow(RESIZED_IMAGE_WINDOW, 0, 0)
            cv2.imshow(IMAGE_INPUT_WINDOW, image)
            cv2.moveWindow(IMAGE_INPUT_WINDOW, 450, 0)
            cv2.imshow(IMAGE_THRESH_WINDOW, thresh)
            cv2.moveWindow(IMAGE_THRESH_WINDOW, 450, 300)
            key = cv2.waitKey(0)
            # Decision for manipulating threshold
            if key == 61:
                threshold += 1
            elif key == 45:
                threshold -= 1
            elif key == 93:
                threshold += 10
            elif key == 91:
                threshold -= 10
            # Confirm threshold
            elif key == 13:
                B, G, R = cv2.split(thresh)
                thresholdOutput = np.where((B == 255) & (G == 255) & (R == 255), 255, 0).astype('uint8')
                cv2.destroyWindow(IMAGE_INPUT_WINDOW)
                cv2.destroyWindow(IMAGE_THRESH_WINDOW)
                break
            print("Current threshold: " + str(threshold))

        print("ENTER to accept\nSPACE to select region to cover\nC to cancel changes")

        # Add image for corrections
        thresholdCorrected = thresholdOutput.copy()
        while True:
            THRESHOLD_OUTPUT_WINDOW = "Threshold output"
            cv2.imshow(THRESHOLD_OUTPUT_WINDOW, thresholdOutput)
            cv2.moveWindow(THRESHOLD_OUTPUT_WINDOW, 450, 0)

            THRESHOLD_CORRECTED_WINDOW = "Corrected threshold"
            cv2.imshow(THRESHOLD_CORRECTED_WINDOW, thresholdCorrected)
            cv2.moveWindow(THRESHOLD_CORRECTED_WINDOW, 450, 300)
            key = cv2.waitKey(0)

            if key == 32:
                thresholdCorrectedResize = cv2.resize(thresholdCorrected, (0, 0), fx=2, fy=2, interpolation=0)
                points = GetPoints(thresholdCorrectedResize)
                cv2.rectangle(thresholdCorrectedResize, points[0], points[1], 0, -1)
                thresholdCorrected = cv2.resize(thresholdCorrectedResize, (0, 0), fx=0.5, fy=0.5, interpolation=0)
            elif key == 99:
                thresholdCorrected = thresholdOutput.copy()
            elif key == 13:
                print("Changes accepted")
                cv2.destroyWindow("select ROI")
                cv2.destroyWindow(THRESHOLD_CORRECTED_WINDOW)
                cv2.destroyWindow(THRESHOLD_OUTPUT_WINDOW)
                break

            cv2.destroyWindow("select ROI")
        cv2.destroyWindow(RESIZED_IMAGE_WINDOW)
        print("ENTER to accept\nSPACE to select region to cover\nC to cancel changes")

        # classify contours

        currentObjectImage = np.zeros(image.shape)
        imagePointer = imageRes.copy()
        cv2.imshow(RESIZED_IMAGE_WINDOW, imagePointer)
        cv2.moveWindow(RESIZED_IMAGE_WINDOW, 0, 0)
        cv2.imshow(LABELED_WINDOW, imageLabeled)
        cv2.moveWindow(LABELED_WINDOW, 450, 0)
        cv2.imshow(MULTIPLIED_WINDOW, imageLabeledMulti)
        cv2.moveWindow(MULTIPLIED_WINDOW, 450, 300)

        contours, _ = cv2.findContours(thresholdCorrected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("1 to save as Crack\n2 to save as Pore\n3 to save as Dirt\n4 to save as Background\nElse to skip")
        for contour in contours:
            imagePointer = imageRes.copy()
            currentObjectImage = np.zeros(image.shape)
            cv2.drawContours(currentObjectImage, [contour], -1, (0, 0, 255), -1).astype('uint8')
            bBox = ResizeBoundingBox(cv2.boundingRect(contour))
            cv2.rectangle(imagePointer, (bBox[0], bBox[1]), (bBox[0] + bBox[2], bBox[1] + bBox[3]), (0, 0, 255), 2)
            cv2.imshow(RESIZED_IMAGE_WINDOW, imagePointer)
            cv2.imshow(CURRENT_OBJECT_WINDOW, currentObjectImage)
            key = cv2.waitKey(0)

            if key == 49:
                cv2.drawContours(imageLabeled, [contour], -1, (1, 1, 1), -1)
                imageLabeledMulti = imageLabeled * 10
                cv2.imshow(LABELED_WINDOW, imageLabeled.astype("uint8"))
                cv2.imshow(MULTIPLIED_WINDOW, imageLabeledMulti.astype('uint8'))
            elif key == 50:
                cv2.drawContours(imageLabeled, [contour], -1, (2, 2, 2), -1)
                imageLabeledMulti = imageLabeled * 10
                cv2.imshow(LABELED_WINDOW, imageLabeled.astype("uint8"))
                cv2.imshow(MULTIPLIED_WINDOW, imageLabeledMulti.astype('uint8'))
            elif key == 51:
                cv2.drawContours(imageLabeled, [contour], -1, (3, 3, 3), -1)
                imageLabeledMulti = imageLabeled * 10
                cv2.imshow(LABELED_WINDOW, imageLabeled.astype("uint8"))
                cv2.imshow(MULTIPLIED_WINDOW, imageLabeledMulti.astype('uint8'))
            elif key == 52:
                cv2.drawContours(imageLabeled, [contour], -1, (4, 4, 4), -1)
                imageLabeledMulti = imageLabeled * 10
                cv2.imshow(LABELED_WINDOW, imageLabeled.astype("uint8"))
                cv2.imshow(MULTIPLIED_WINDOW, imageLabeledMulti.astype('uint8'))
            else:
                print("Skipped")

        cv2.imshow(BREAK_IMAGE, breakImg)
        key = cv2.waitKey(0)
        if key == 49:
            print("Updating image")
            cv2.destroyAllWindows()
        else:
            print("Loading next image")
            cv2.destroyAllWindows()
            # Move existing image
            os.rename(IMAGES_DIR + "/" + os.path.basename(inputPaths[i]), IMAGES_DONE_DIR + "/" + os.path.basename(inputPaths[i]))
            cv2.imwrite(OUTPUT_DIR_IMAGES + "/" + os.path.basename(inputPaths[i])[:-3] + "png",
                        image.astype("uint8"))
            cv2.imwrite(OUTPUT_DIR_LABELS + "/" + os.path.basename(inputPaths[i])[:-3] + "png",
                        imageLabeled.astype("uint8"))
            break
