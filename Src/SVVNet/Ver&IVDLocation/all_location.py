import sys

sys.setrecursionlimit(10000)
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# 4邻域的连通域和 8邻域的连通域
# [row, col]
NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]

NEIGHBOR_HOODS_8 = False
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1, 0], [0, 0], [1, 0],
             [-1, 1], [0, 1], [1, 1]]


def reorganize(binary_img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num
            points[index].append([row, col])
    return binary_img, points


def neighbor_value(binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows - 1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols - 1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row + offset[0]), rows - 1)
                neighbor_col = min(max(0, col + offset[1]), cols - 1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    return binary_img


# binary_img: bg-0, object-255; int
def Two_Pass(binary_img: np.array, neighbor_hoods):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    binary_img = neighbor_value(binary_img, offsets, False)
    binary_img = neighbor_value(binary_img, offsets, True)

    return binary_img


def recursive_seed(binary_img: np.array, seed_row, seed_col, offsets, num, max_num=1000):
    rows, cols = binary_img.shape
    binary_img[seed_row][seed_col] = num
    for offset in offsets:
        neighbor_row = min(max(0, seed_row + offset[0]), rows - 1)
        neighbor_col = min(max(0, seed_col + offset[1]), cols - 1)
        var = binary_img[neighbor_row][neighbor_col]
        if var < max_num:
            continue
        binary_img = recursive_seed(binary_img, neighbor_row, neighbor_col, offsets, num, max_num)
    return binary_img


# max_num 表示连通域最多存在的个数
def Seed_Filling(binary_img, neighbor_hoods, max_num=100):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    num = 1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var <= max_num:
                continue
            binary_img = recursive_seed(binary_img, row, col, offsets, num, max_num=100)
            num += 1
    return binary_img


if __name__ == "__main__":
    img_names = os.listdir('../outputs/testdata/')[:-1]
    w = 16
    h = 24
    addIVDs = 3
    addVer = 2
    var = addIVDs - addVer
    hashtableHeat = np.load('../outputs/testdata/LocPos.npy', allow_pickle=True).item()

    train_img_names, val_img_names = train_test_split(img_names, test_size=0.2, random_state=40)

    #    img_names =  ["Case65"]
    for index in range(len(img_names)):

        #######################################################################
        # Ivds
        #######################################################################
        binary_img = cv2.imread("../outputs/testdata/{}/PredIVDs.bmp".format(img_names[index]), cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread("../outputs/testdata/{}/PredIVDs.bmp".format(img_names[index]), cv2.IMREAD_GRAYSCALE)

        binary_img = Seed_Filling(binary_img, NEIGHBOR_HOODS_8)
        binary_img, points = reorganize(binary_img)

        num = 0
        d = []
        for i in range(len(points)):
            p = np.array(points[i])
            pmean = p.mean(axis=0)
            if len(points[i]) > 80 or (pmean[0] > 235 and len(points[i]) > 50):
                if i == 0:
                    xmin, ymin = points[0][0]
                    xmax, ymax = points[0][-1]
                    a = np.array(points[0])
                    b = np.array(points[1])
                    if np.mean(a[:, 1]) - np.mean(b[:, 1]) > 15:
                        continue
                    #                    print(np.mean(a[:,1]) - np.mean(b[:,1]))
                    if len(points[0]) < len(points[1]) - 150 and np.mean(a[:, 1]) - np.mean(b[:, 1]) > 10 and len(
                            points[0]) < 120:
                        continue
                num += 1
                xmean = 0
                ymean = 0
                for j in range(len(points[i])):
                    x, y = points[i][j][0], points[i][j][1]
                    xmean += x
                    ymean += y
                    binary_img[x, y] = 255
                d.append(
                    [xmean / len(points[i]) + addIVDs, ymean / len(points[i]) + hashtableHeat[img_names[index]] - 64])
                xmean //= len(points[i])
                ymean //= len(points[i])
                cv2.rectangle(mask_img, (max(0, ymean - h), max(0, xmean - w)),
                              (min(128, ymean + h), min(256, xmean + w)), 255, 2)
        print("IVDs:{}".format(num))
        save = pd.DataFrame(d, columns=['x', 'y'])
        cv2.imwrite("../outputs/testdata/{}/PredIVDs-Box.bmp".format(img_names[index]), mask_img)
        save.to_csv("../outputs/testdata/{}/PredPoints.csv".format(img_names[index]), index=True,
                    header=True)  # index=False,header=False表示不保存行索引和列标题

        #######################################################################
        # Ver
        #######################################################################
        ver_binary_img = cv2.imread("../outputs/testdata/{}/PredVer.bmp".format(img_names[index]), cv2.IMREAD_GRAYSCALE)
        ver_mask_img = cv2.imread("../outputs/testdata/{}/PredVer.bmp".format(img_names[index]), cv2.IMREAD_GRAYSCALE)

        num = 0
        dVer = []
        for i in range(len(d) + 1):
            if i == 0:
                temp = ver_binary_img[:int(d[i][0]) - addIVDs, :]
                add = 0
            elif i == len(d):
                temp = ver_binary_img[int(d[i - 1][0]) - addIVDs:, :]
                add = int(d[i - 1][0])
            else:
                temp = ver_binary_img[int(d[i - 1][0]) - addIVDs:int(d[i][0]) - addIVDs, :]
                add = int(d[i - 1][0])
            pointNums = 0
            xmean = 0
            ymean = 0
            for px in range(len(temp)):
                for py in range(len(temp[0])):
                    if temp[px][py] == 255:
                        xmean += px
                        ymean += py
                        pointNums += 1
            if pointNums != 0 and pointNums > 150:
                num += 1
                dVer.append(
                    [xmean / pointNums + addVer + add, ymean / pointNums + hashtableHeat[img_names[index]] - 64])
                xmean //= pointNums
                ymean //= pointNums
                xmean += add
            cv2.rectangle(ver_mask_img, (max(0, int(ymean - 2 * h)), max(0, int(xmean - 2 * w))),
                          (min(128, int(ymean + 2 * h)), min(256, int(xmean + 2 * w))), 128, 2)
        save = pd.DataFrame(dVer, columns=['x', 'y'])
        cv2.imwrite("../outputs/testdata/{}/PredVer-Box.bmp".format(img_names[index]), ver_mask_img)
        save.to_csv("../outputs/testdata/{}/PredPointsVer.csv".format(img_names[index]), index=True,
                    header=True)  # index=False,header=False表示不保存行索引和列标题