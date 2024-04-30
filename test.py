import cv2
import numpy as np
from matplotlib import pyplot as plt

gEasyOne = cv2.imread("dataset/ground_truths/easy/easy_1.png")
gEasyTwo = cv2.imread("dataset/ground_truths/easy/easy_2.png")
gEasyThree = cv2.imread("dataset/ground_truths/easy/easy_3.png")

gMediumOne = cv2.imread("dataset/ground_truths/medium/medium_1.png")
gMediumTwo = cv2.imread("dataset/ground_truths/medium/medium_2.png")
gMediumThree = cv2.imread("dataset/ground_truths/medium/medium_3.png")

gHardOne = cv2.imread("dataset/ground_truths/hard/hard_1.png")
gHardTwo = cv2.imread("dataset/ground_truths/hard/hard_2.png")
gHardThree = cv2.imread("dataset/ground_truths/hard/hard_3.png")

easyOne = cv2.imread("dataset/output/easy/easy_1.jpg", cv2.IMREAD_GRAYSCALE)
easyTwo = cv2.imread("dataset/output/easy/easy_2.jpg", cv2.IMREAD_GRAYSCALE)
easyThree = cv2.imread("dataset/output/easy/easy_3.jpg", cv2.IMREAD_GRAYSCALE)

mediumOne = cv2.imread("dataset/output/medium/medium_1.jpg", cv2.IMREAD_GRAYSCALE)
mediumTwo = cv2.imread("dataset/output/medium/medium_2.jpg", cv2.IMREAD_GRAYSCALE)
mediumThree = cv2.imread("dataset/output/medium/medium_3.jpg", cv2.IMREAD_GRAYSCALE)

hardOne = cv2.imread("dataset/output/hard/hard_1.jpg", cv2.IMREAD_GRAYSCALE)
hardTwo = cv2.imread("dataset/output/hard/hard_2.jpg", cv2.IMREAD_GRAYSCALE)
hardThree = cv2.imread("dataset/output/hard/hard_3.jpg", cv2.IMREAD_GRAYSCALE)


def tester(truth, image):
    outArray = []

    # truth should be BGR
    truthflatten = truth.reshape(-1, truth.shape[-1]);  # truthflatten should have shape (x, [])

    # image should be black and white, foreground value should be 255
    flatten = image.flatten();  # flatten should have the shape (x)

    truepos = 0;
    falsepos = 0;
    trueneg = 0;
    falseneg = 0;

    i = 0;
    while i < image.size:
        truthpixel = truthflatten[i];
        if flatten[i] == 255:  # Positive
            if np.array_equal(truthpixel, [0, 0, 128]) or np.array_equal(truthpixel, [0, 0, 0]):
                truepos += 1;
            else:
                falsepos += 1;
        else:  # Negative
            if np.array_equal(truthpixel, [0, 128, 128]) or np.array_equal(truthpixel, [0, 128, 0]) or np.array_equal(
                    truthpixel, [0, 0, 0]):
                trueneg += 1;
            else:
                falseneg += 1;
        i += 1;

    print(truepos)
    print(falsepos)
    print(trueneg)
    print(falseneg)

    accuracy = (truepos + trueneg) / (truepos + falsepos + trueneg + falseneg);
    print(accuracy)
    outArray.append(accuracy)
    precision = (truepos) / (truepos + falsepos)
    print(precision)
    outArray.append(precision)
    recall = (truepos) / (truepos + falseneg)
    print(recall)
    outArray.append(recall)
    iou = (truepos) / (truepos + falsepos + falseneg)
    print(iou)
    outArray.append(iou)
    dice = 2 / ((1 / iou) + 1)
    print(dice)
    outArray.append(dice)
    return outArray


print("Easy 1")
easy1 = tester(gEasyOne, easyOne)
print("Easy 2")
easy2 = tester(gEasyTwo, easyTwo)
print("Easy 3")
easy3 = tester(gEasyThree, easyThree)
print("medium 1")
medium1 = tester(gMediumOne, mediumOne)
print("medium 2")
medium2 = tester(gMediumTwo, mediumTwo)
print("medium 3")
medium3 = tester(gMediumThree, mediumThree)
print("hard 1")
hard1 = tester(gHardOne, hardOne)
print("hard 2")
hard2 = tester(gHardTwo, hardTwo)
print("hard 3")
hard3 = tester(gHardThree, hardThree)

avgAcc = (easy1[0] + easy2[0] + easy3[0] + medium1[0] + medium2[0] + medium3[0] + hard1[0] + hard2[0] + hard3[0])/9
avgPre = (easy1[1] + easy2[1] + easy3[1] + medium1[1] + medium2[1] + medium3[1] + hard1[1] + hard2[1] + hard3[1])/9
avgRec = (easy1[2] + easy2[2] + easy3[2] + medium1[2] + medium2[2] + medium3[2] + hard1[2] + hard2[2] + hard3[2])/9
avgIOU = (easy1[3] + easy2[3] + easy3[3] + medium1[3] + medium2[3] + medium3[3] + hard1[3] + hard2[3] + hard3[3])/9
avgDice = (easy1[4] + easy2[4] + easy3[4] + medium1[4] + medium2[4] + medium3[4] + hard1[4] + hard2[4] + hard3[4])/9
print("average accuracy = " + str(avgAcc))
print("average precision = " + str(avgPre))
print("average recall = " + str(avgRec))
print("average iou = " + str(avgIOU))
print("average dice = " + str(avgDice))
