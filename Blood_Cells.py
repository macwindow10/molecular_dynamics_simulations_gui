from __future__ import print_function

import PIL.Image
import PIL.ImageTk
import tkinter
import pyautogui
import math
import numpy as np
import cv2
from vidstab import VidStab
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from math import sqrt
import glob
import os
from multiprocessing.pool import ThreadPool
from collections import deque
import tkinter as tk
from tkinter import filedialog
import time
from common import clock, StatValue, draw_str, RectSelector
import video
import warnings

warnings.filterwarnings('ignore')


class DummyTask:
    def __init__(self, data):
        self.data = data

    def ready(self):
        return True

    def get(self):
        return self.data


class App:
    # <! For Medaka Trackbar control
    def medaka_trackbar_callback(self, value):
        if value == 0:
            value = 1
        self.ratio = value * 0.2

    # ----------------------------------!>
    # <! For ZebraFish Trackbar control
    def zebrafish_trackbar_callback(self, value):
        if value == 0:
            value = 1
        self.ratio = value * 0.6

    # ----------------------------------!>

    def __init__(self, canvas, propotion, video_src, roiwid, graph_path, roihei, type, is_selected, paused=False):
        print("app init entered", paused)
        self.type = type
        self.canvas = canvas
        self.filename = video_src.split('\\')[2].split('.')[0]
        # self.filename = video_src.split('/')[-1].split('.')[0]
        print(video_src.split('\\')[2].split('.'))
        # print('video src',video_src.split('/')[-1].split('.')[0])

        self.path = graph_path
        self.cap = video.create_capture(video_src)
        self.frameLength = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        _, self.base_frame = self.cap.read()
        cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
        cv2.imshow('Select ROI', self.base_frame)
        # self.rect_sel = RectSelector('Select ROI', self.onrect, roiwid, roihei)
        self.rect_sel = RectSelector('Select ROI', self.onrect)

        self.croped_pos = (0, 0, self.base_frame.shape[1], self.base_frame.shape[0])
        self.testNo = 0

        self.rePropotion = propotion
        self.roiWidth = roiwid
        self.paused = paused
        self.latency = StatValue()
        self.frame_interval = StatValue()
        self.last_frame_time = clock()
        self.is_croped = False
        self.is_selected = is_selected  # False manual True Automatic
        self.manual = ~is_selected
        self.cnt_find = 0
        self.count_one_sec_list = []

        self.init_value()
        self.init_thread()

    def init_thread(self):
        self.stabilizer = VidStab()
        self.threadn = cv2.getNumberOfCPUs()
        self.pool = ThreadPool(processes=self.threadn)
        self.pending = deque()
        self.threaded_mode = False
        self.pressESC = False

    def init_value(self):
        self.init_flow = True
        self.isFindbloodLine = False
        self.isCanDetectCellsCount = False

        self.prev_frame = None
        self.opt_flow = None
        self.old_bloodLine = None
        self.mask_blood = None
        self.result_line = None
        self.avarageCellSize = None

        self.count_folow_detect = 0
        self.frameNo = 0
        self.averageSpeed = 0
        self.averageCount = 0

        self.pointArray = []
        self.cellsSpeeds = []
        self.cellsCounts = []
        self.lstSpeeds = []
        self.lstCounts = []
        self.completeROI = False

    def process_frame(self, gFrame, t0, scale):
        prev_frame = np.copy(self.prev_frame)
        if self.isFindbloodLine:
            mask = cv2.resize(self.result_line, (0, 0), None, scale, scale)
            mask = np.ceil(mask / 255).astype('uint8')
            gFrame = gFrame * mask
            prev_frame = prev_frame * mask

        if self.init_flow:
            opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gFrame, None, 0.5, 5, 13, 10, 5, 1.1,
                                                    cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
            self.init_flow = False
        else:
            if (np.sum(self.opt_flow) is not None):
                opt_flow = cv2.resize(self.opt_flow, (0, 0), None, scale, scale)
                opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gFrame, opt_flow, 0.5, 5, 13, 10, 5, 1.1,
                                                        cv2.OPTFLOW_USE_INITIAL_FLOW)
            else:
                opt_flow = cv2.calcOpticalFlowFarneback(prev_frame, gFrame, None, 0.5, 5, 13, 10, 5, 1.1,
                                                        cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
                self.init_flow = False
        self.opt_flow = cv2.resize(opt_flow, (0, 0), None, 1 / scale, 1 / scale)

        return t0

    def GetROI(self, img):
        thresh, bimg = cv2.threshold(img, 30, 255, 0)
        contours, hierarchy = cv2.findContours(bimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # tcolor = img[0,0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 600:
                # cv2.fillPoly(img,pts=cnt,color=(tt,tt,tt))
                cv2.fillPoly(img, pts=[cnt], color=(11, 11, 11))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sy = -1
        ey = -1
        # for j in range(0,img.shape[1]):
        #     print(img.item(0, j))
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                pixel = img.item(i, j)
                if pixel > 30:
                    if sy < 0:
                        sy = i;
                    break;
            if j == (img.shape[1] - 1) and sy > 0:
                ey = i
                break;
        # img = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        sx = ex = -1
        for j in range(0, img.shape[1]):
            for i in range(sy, ey):
                pixel = img.item(i, j)
                if pixel > 30:
                    sx = j;
                    break;
            if i < (ey - 1):
                break;

        for j in range(0, img.shape[1]):
            vj = img.shape[1] - j - 1
            for i in range(sy, ey):
                pixel = img.item(i, vj)
                if pixel > 30:
                    ex = vj;
                    break;
            if i < (ey - 1):
                break;
        # img = cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
        self.croped_pos = (sx, sy, ex, ey)

        # cnt = contours[0]

    #     # print(x, y, w, h)
    # def GetROI(self, img):
    #     cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = cnts[0] if len(cnts) == 2 else cnts[1]
    #     cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    #     # cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     # cnt = max(cnts, key=cv2.contourArea)
    #     cnt = cntsSorted[1]
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     self.croped_pos = (x, y, w, h)
    def cubic_interp1d(self, x0, x, y):
        """
        Interpolate a 1-D function using cubic splines.
        x0 : a float or an 1d-array
        x : (N,) array_like
            A 1-D array of real/complex values.
        y : (N,) array_like
            A 1-D array of real values. The length of y along the
            interpolation axis must be equal to the length of x.

        Implement a trick to generate at first step the cholesky matrice L of
        the tridiagonal matrice A (thus L is a bidiagonal matrice that
        can be solved in two distinct loops).

        additional ref: www.math.uh.edu/~jingqiu/math4364/spline.pdf 
        """
        x = np.asfarray(x)
        y = np.asfarray(y)

        # remove non finite values
        # indexes = np.isfinite(x)
        # x = x[indexes]
        # y = y[indexes]

        # check if sorted
        if np.any(np.diff(x) < 0):
            indexes = np.argsort(x)
            x = x[indexes]
            y = y[indexes]

        size = len(x)

        xdiff = np.diff(x)
        ydiff = np.diff(y)

        # allocate buffer matrices
        Li = np.empty(size)
        Li_1 = np.empty(size - 1)
        z = np.empty(size)

        # fill diagonals Li and Li-1 and solve [L][y] = [B]
        Li[0] = sqrt(2 * xdiff[0])
        Li_1[0] = 0.0
        B0 = 0.0  # natural boundary
        z[0] = B0 / Li[0]

        for i in range(1, size - 1, 1):
            Li_1[i] = xdiff[i - 1] / Li[i - 1]
            Li[i] = sqrt(2 * (xdiff[i - 1] + xdiff[i]) - Li_1[i - 1] * Li_1[i - 1])
            Bi = 6 * (ydiff[i] / xdiff[i] - ydiff[i - 1] / xdiff[i - 1])
            z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

        i = size - 1
        Li_1[i - 1] = xdiff[-1] / Li[i - 1]
        Li[i] = sqrt(2 * xdiff[-1] - Li_1[i - 1] * Li_1[i - 1])
        Bi = 0.0  # natural boundary
        z[i] = (Bi - Li_1[i - 1] * z[i - 1]) / Li[i]

        # solve [L.T][x] = [y]
        i = size - 1
        z[i] = z[i] / Li[i]
        for i in range(size - 2, -1, -1):
            z[i] = (z[i] - Li_1[i - 1] * z[i + 1]) / Li[i]

        # find index
        index = x.searchsorted(x0)
        np.clip(index, 1, size - 1, index)

        xi1, xi0 = x[index], x[index - 1]
        yi1, yi0 = y[index], y[index - 1]
        zi1, zi0 = z[index], z[index - 1]
        hi1 = xi1 - xi0

        # calculate cubic
        f0 = zi0 / (6 * hi1) * (xi1 - x0) ** 3 + \
             zi1 / (6 * hi1) * (x0 - xi0) ** 3 + \
             (yi1 / hi1 - zi1 * hi1 / 6) * (x0 - xi0) + \
             (yi0 / hi1 - zi0 * hi1 / 6) * (xi1 - x0)
        return f0

    # <! Make TitleMask for stablize video
    def makeTitleMask(self, frame, proportion=0.5):
        frame = cv2.resize(frame, (0, 0), None, proportion, proportion)
        mask = np.zeros_like(frame)
        shape = np.shape(mask)
        pt1 = (int(2 * shape[1] / 100), int(3 * shape[0] / 100))
        pt2 = (int(98 * shape[1] / 100), int(97 * shape[0] / 100))
        cv2.rectangle(mask, pt1, pt2, (255, 255, 255), -1)
        mask = np.divide(mask, 255).astype('uint8')
        return mask

    # --------------------------------------------------------!>

    # <! Filter Any Frame
    # Equalize Histograme
    # Calhe image

    def filterProcess(self, gray):
        equ_img = cv2.equalizeHist(gray)
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(50, 50))
        cl1 = clahe.apply(equ_img)
        return cl1

    # ----------------------------------------------------!>

    # <! Crop Image by Custome Rect
    def CropImgByRect(self, img, rect):
        if not rect:
            return None
        x1, y1, x2, y2 = rect
        w, h = map(cv2.getOptimalDFTSize, [x2 - x1, y2 - y1])
        x1, y1 = (x1 + x2 - w) // 2, (y1 + y2 - h) // 2
        x, y = x1 + 0.5 * (w - 1), y1 + 0.5 * (h - 1)
        area = cv2.getRectSubPix(img, (max(w, 1), max(h, 1)), (x, y))
        return area

    # --------------------------------------------------------------!>

    # <! Is Custome Point  Exist in Rect
    def IsExistPoint(self, point, croped_pos):
        if point[0][0] > croped_pos[1] and point[0][0] < croped_pos[3]:
            if point[0][1] > croped_pos[0] and point[0][1] < croped_pos[2]:
                return True
        return False

    # -------------------------------------------------------------------------!>

    def AvarageFilter(self, lst, filter_size, total_size):
        i = 0
        lst_value = np.copy(lst)
        sizelst = np.size(lst_value)
        average = np.average(lst)

        scale = int(total_size / sizelst)
        fscale = float(total_size / sizelst)
        other = total_size % sizelst

        if (scale > 1) or (other > 0):
            result = np.zeros(sizelst * scale, dtype=float)
            result[::scale] = lst_value
            zeros = np.zeros(other, dtype=float)
            result = np.append(result, zeros)
        else:
            result = lst_value

        newlist = np.zeros(total_size, dtype=float)

        i = 0
        for value in newlist:
            idx = int(i / fscale)
            newlist[i] = lst_value[idx]
            i += 1
        i = 0
        finallist = np.zeros(total_size, dtype=float)
        average_stand = 40
        for value in newlist:
            if (i < total_size - average_stand):
                finallist[i] = np.average(newlist[i:i + average_stand])
            else:
                finallist[i] = average
            i += 1

        return finallist

        i = 0
        for value in result:
            if (value == 0):
                result[i] = average
            i += 1
        sizelst = np.size(result)
        i = 0
        for value in result:
            if (i > filter_size) and (i < (sizelst - filter_size)):
                result[i] = np.average(result[i - filter_size: i + filter_size])
            else:
                if (i <= filter_size):
                    result[i] = np.average(result[i: i + filter_size])
                else:
                    # if(i >=(sizelst- filter_size)) and i < sizelst:
                    if (i >= (sizelst - filter_size)):
                        result[i] = np.average(result[i - filter_size: i])
            i += 1
        return result

    # <! Filter object by Size
    # img : source image
    # filter_1 : first fiter value
    # filter_2 : second filter value
    def filterSize(self, img, filter_1, filter_2):
        img = img.astype('uint8')
        # find contours%
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # filter out contours by size
        small_cntrs = []
        width_list = []
        for con in contours:
            area = cv2.contourArea(con)
            #print(cv2.minAreaRect(con))
            width_list.append(cv2.minAreaRect(con)[1][0])
            self.width = np.average(width_list)
            # print(area)
            if filter_1 < area < filter_2:  # size threshold
                small_cntrs.append(con)
        return small_cntrs

    # ------------------------------------------------------------------!>

    # <! Get count of cells by proportion feature
    def GetAverageCount(self, img, filter_1=10, filter_2=15):
        contours = self.filterSize(img, 100, 200)
        small_cntrs = []
        for con in contours:
            shape = np.shape(con)
            proportion = shape[0] / shape[2]
            # print("THE PROPORTION IS")
            # print(proportion)
            # print(shape)
            # print(con)
            if filter_1 < proportion < filter_2:
                small_cntrs.append(cv2.contourArea(con))
        if (np.size(small_cntrs) > 0):
            return np.average(small_cntrs)
        else:
            return -1

    # -----------------------------------------------------------!>

    # <!Calculate Speed of Cells
    def display_flow(self, flow, stride=40):
        # <Calculate Speed-------------------------
        deltaArray = []
        pointArray = []
        for index in np.ndindex(flow[::stride, ::stride].shape[:2]):
            pt1 = tuple(i * stride for i in index)
            delta = flow[pt1].astype(np.int32)[::-1]
            if 1 <= cv2.norm(delta) <= 7:
                delta = np.absolute(delta)
                pt2 = tuple(pt1 + 5 * delta)
                deltaArray.append(delta)
                pointArray.append((pt1, pt2))
        return (np.ceil(deltaArray), pointArray)

    # --------------------------------------------!>

    # <! Finish select rect
    def onrect(self, rect):
        print("onrect enterd")
        self.croped_pos = rect
        self.is_selected = True

    # ---------------------------------!>

    # <! Draw Graph
    def drawGraph(self, x, y, x_new, avarage, y_lab, title):
        fig, ax = plt.subplots()
        cubic = self.cubic_interp1d(x_new, x, y)
        # ax.plot(x_new, cubic, color='blue', linewidth=1)
        ax.plot(x_new, cubic, color='C1', linewidth=1)

        ax.set(xlabel='frames (30/s)', ylabel=y_lab, title=title)
        # ax.set(xlabel='frames (30/s)', ylabel = y_lab)
        ax.grid()
        fig.canvas.draw()
        avarageListArr = np.arange(np.size(x_new))
        y2 = []
        for i in avarageListArr:
            y2.append(avarage)

        ax.plot(x_new, y2, color='crimson')
        fig.canvas.draw()

        graphImg = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        # graphImg  = graphImg.reshape((480,1000) + (3,))
        graphImg = graphImg.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        graphImg = cv2.cvtColor(graphImg, cv2.COLOR_RGB2BGR)
        return graphImg

    # -----------------------------------------------------------------!>

    def run(self):
        width = int(np.shape(self.base_frame)[1] * self.rePropotion)
        height = int(np.shape(self.base_frame)[0] * self.rePropotion)
        cv2.resizeWindow('Select ROI', width, height + 40)
        if self.type == '1':
            cv2.createTrackbar('Medaka', 'Select ROI', 8, 10, lambda v: self.medaka_trackbar_callback(v))
        else:
            cv2.createTrackbar('ZebraFish', 'Select ROI', 3, 10, lambda v: self.zebrafish_trackbar_callback(v))

        grabbed_frame, frame = self.cap.read()
        title_mask = self.makeTitleMask(frame, self.rePropotion)
        my_count = 0
        start = time.time()
        self.count_one_sec = 0
        self.count_one_sec_calculation = 0
        while True:
            t = clock()
            self.frame_interval.update(t - self.last_frame_time)
            self.last_frame_time = t

            # while len(self.pending) > 0 and self.pending[0].ready():
            while len(self.pending) > 0 and self.pending[0].ready():
                my_count += 1
                t0 = self.pending.popleft().get()
                gFrame = cv2.cvtColor(self.base_frame, cv2.COLOR_BGR2GRAY)
                self.prev_frame = cv2.resize(gFrame, (0, 0), None, self.rePropotion, self.rePropotion)
                flowResult = self.display_flow(self.opt_flow)
                deltaArray = flowResult[0]
                self.pointArray = flowResult[1]
                if self.isFindbloodLine == False:

                    self.cnt_find = self.cnt_find + 1;
                    if self.cnt_find == 30:
                        # exit();
                        return
                    norm_opt_flow = np.linalg.norm(self.opt_flow, axis=2)
                    norm_opt_flow = cv2.normalize(norm_opt_flow, None, 0, 1, cv2.NORM_MINMAX)

                    ret, thresh_img = cv2.threshold(norm_opt_flow, 0.19, 255, cv2.THRESH_BINARY_INV)
                    shape = np.shape(self.base_frame)
                    fiter_size = int((shape[0] * shape[1]) / (25 * 25))
                    small_cntrs = self.filterSize(thresh_img, 1, fiter_size)
                    colorImg = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
                    cv2.drawContours(colorImg, small_cntrs, -1, (255, 255, 255), -1)

                    mask_img = (colorImg - 255) / -255
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                    if self.old_bloodLine is not None:
                        if np.sum(mask_img) > int((shape[0] * shape[1]) / (25 * 25)):
                            diffBloodLine = np.diff(mask_img - self.old_bloodLine)
                            sum = np.sum(diffBloodLine)
                            if sum < fiter_size / 10:
                                self.count_folow_detect = self.count_folow_detect + 1
                                if self.result_line is not None:
                                    self.result_line = self.result_line + self.old_bloodLine
                                else:
                                    self.result_line = self.old_bloodLine
                            else:
                                self.count_folow_detect = 0
                    self.old_bloodLine = mask_img

                    if self.count_folow_detect > 5:
                        ret, self.result_line = cv2.threshold(self.result_line, 1, 255, cv2.THRESH_BINARY)
                        self.mask_blood = np.ceil(self.result_line / 255)
                        self.isFindbloodLine = True
                else:
                    mask = self.mask_blood.astype('uint8')
                    filter_result = self.filterProcess(gFrame * mask)

                    if self.manual == -2:
                        self.GetROI(filter_result)

                    # my adding start
                    tx, ty = filter_result.shape
                    xs, ys, xe, ye = self.croped_pos
                    # temp=np.ones((tx,ty)) * 11
                    # temp[xs:xe-1,ys:ye-1] = filter_result[xs:xe-1,ys:ye-1]
                    # filter_result = temp

                    if self.manual == -2:
                        for i in range(tx):
                            for j in range(ty):
                                if i > ys and i < ye and j > xs and j < xe:

                                    continue
                                else:

                                    filter_result[i, j] = 11

                    eroded = cv2.morphologyEx(filter_result, cv2.MORPH_ERODE, (3, 3), iterations=3)
                    # cv2.imshow('nothing',eroded)
                    # cv2.waitKey(0)
                    # self.GetROI(eroded)

                    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
                    ret, thresh_img = cv2.threshold(closed, 100, 255, cv2.THRESH_BINARY_INV)
                    if self.isCanDetectCellsCount == False:
                        self.avarageCellSize = self.GetAverageCount(thresh_img)
                        if (self.avarageCellSize > 0):
                            self.isCanDetectCellsCount = True
                    else:

                        totalNum = np.sum(self.CropImgByRect(self.mask_blood, self.croped_pos))
                        selected_area = self.CropImgByRect(thresh_img, self.croped_pos)
                        black = np.size(selected_area) - np.sum(np.ceil(selected_area / 255))
                        white = totalNum - black
                        cellCount = np.ceil(white / self.avarageCellSize)
                        # cv2.imshow("test", selected_area)
                        if (np.size(deltaArray) == 0):
                            speed = 0.1
                        else:
                            speed = np.average(deltaArray)
                        if self.frameNo % 3 != 0:
                            self.cellsSpeeds.append(speed)
                            self.cellsCounts.append(cellCount)
                        else:
                            self.averageSpeed = np.average(self.cellsSpeeds)
                            self.averageCount = np.average(self.cellsCounts)
                            self.cellsSpeeds = []
                            self.cellsCounts = []
                        x1, y1, x2, y2 = self.croped_pos
                        avg_cell_width = self.width * (25.4 / 96)
                        avg_blood_cells_per_total_width = ((x2 - x1) * (25.4 / 96)) / (self.width * (25.4 / 96))
                        avg_blood_cells_rows = self.averageCount / avg_blood_cells_per_total_width
                        speed_per_frame = (self.averageSpeed / self.ratio) / self.frame_interval.value
                        if self.count_one_sec_calculation == 0:
                            self.count_one_sec_calculation = self.averageCount
                        if time.time() - start > 1:
                            self.count_one_sec = self.count_one_sec_calculation
                            self.count_one_sec_list.append(self.count_one_sec_calculation)
                            self.count_one_sec_calculation = 0
                            start = time.time()
                        else:
                            self.count_one_sec_calculation += (avg_blood_cells_rows * avg_cell_width / speed_per_frame)
                        if np.isnan(self.count_one_sec):
                            self.count_one_sec = 0
                        text = 'Average Speed : {:.2f} mm / s, Cells Count : {:.0f}'.format(
                            self.averageSpeed / self.ratio, self.averageCount)
                        if (math.isnan(self.averageSpeed)):
                            self.averageSpeed = 0
                        if (math.isnan(self.averageCount)):
                            self.averageCount = 0
                        self.lstSpeeds.append(self.averageSpeed)
                        self.lstCounts.append(self.averageCount)
                        if self.threaded_mode:
                            threadState = "Multi Thread Process"
                        else:
                            threadState = "Single Thread Process"
                        print_result = cv2.cvtColor(filter_result, cv2.COLOR_GRAY2BGR)
                        cv2.putText(print_result, text, (20, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1);

                        draw_str(print_result, (20, 460), threadState)
                        draw_str(print_result, (20, 480),
                                 "frame interval :  %.1f ms" % (self.frame_interval.value * 1000))
                        x1, y1, x2, y2 = self.croped_pos
                        draw_str(print_result, (20, 500), "ROI width :  %dmm, height : %dmm" % (
                            (x2 - x1) * (25.4 / 96), (y2 - y1) * (25.4 / 96)))
                        cv2.rectangle(print_result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        for point in self.pointArray:
                            if self.IsExistPoint(point, self.croped_pos):
                                result = [point[0][::-1], point[1][::-1]]
                                cv2.arrowedLine(print_result, result[0], result[1], (0, 0, 255), 1, cv2.LINE_8, 0, 0.4)
                        # cv2.imshow('ROI Detect', print_result)

                        # ASIF
                        # now = datetime.now()
                        # dt_string = now.strftime("%Y%m%d_%H%M%S.jpg")
                        # print("Today's date:", dt_string)
                        # cv2.imwrite(dt_string, print_result)
                        try:
                            frame_gui = cv2.resize(print_result, (650, 500))
                            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_gui))
                            self.canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
                            self.canvas.update()
                        except Exception as e:
                            print(e)

                self.latency.update(clock() - t0)
                self.frameNo += 1
            if len(self.pending) < self.threadn:

                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (0, 0), None, self.rePropotion, self.rePropotion)

                if (self.is_croped == False):

                    frame = cv2.resize(self.base_frame, (0, 0), None, self.rePropotion, self.rePropotion)
                    ################
                    self.rect_sel.draw(frame)
                    ####################

                    if self.is_selected:
                        x0, y0, x1, y1 = self.croped_pos
                        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.putText(frame, "Press SPACE key for next step!", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.7,
                                    (0, 0, 255), 1);
                        print('rt')
                    else:
                        cv2.putText(frame, "Select ROI By MouseDroping", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.7,
                                    (0, 0, 255), 1);

                    cv2.imshow('Select ROI', frame)

                    ch = cv2.waitKey(5)

                    if ch == -1 and self.is_selected:
                        cv2.destroyWindow('Select ROI')
                        self.init_value()
                        self.init_thread()
                        self.is_croped = True
                        # cv2.namedWindow('ROI Detect', cv2.WINDOW_NORMAL)
                        # cv2.namedWindow('Base Frame', cv2.WINDOW_NORMAL)
                        width = int(np.shape(self.base_frame)[1] * self.rePropotion)
                        height = int(np.shape(self.base_frame)[0] * self.rePropotion)
                        # cv2.resizeWindow('Base Frame', width, height)
                        # cv2.resizeWindow('ROI Detect', width, height)
                else:

                    # Stablize each frame
                    # stabilized_frame will be an all black frame until iteration 30
                    stabilized_frame = self.stabilizer.stabilize_frame(input_frame=frame, smoothing_window=10)

                    if np.max(stabilized_frame) > 0:
                        self.base_frame = np.multiply(stabilized_frame, title_mask)
                    else:
                        self.base_frame = np.multiply(frame, title_mask)

                    vis = self.base_frame.copy()

                    self.rect_sel.print(vis)
                    process_img = self.process_img = cv2.resize(self.base_frame, (0, 0), None, self.rePropotion,
                                                                self.rePropotion)
                    # <! If User Selected ROI Area
                    if process_img is not None:
                        gFrame = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)
                        if self.prev_frame is None:
                            self.prev_frame = np.copy(gFrame)

                        if self.threaded_mode:
                            task = self.pool.apply_async(self.process_frame, (gFrame, t, self.rePropotion))
                        else:
                            task = DummyTask(self.process_frame(gFrame, t, self.rePropotion))
                        self.pending.append(task)

                    # -------------------------------------------------------------------------!>
                    result_img = vis
                    shape = np.shape(result_img);
                    # cv2.resizeWindow('Base Frame', shape[1], shape[0])
                    # cv2.imshow('Base Frame', result_img)
                    self.testNo += 1

            ch = cv2.waitKey(1)

            scale = 200
            global escape_pressed
            if ch == 27 or escape_pressed:
                self.pressESC = True
                break
            else:
                scale = 1000

        self.cap.release()

        if (self.pressESC):
            totalXcount = self.frameNo;
        else:
            totalXcount = self.frameLength;

        s_width, s_height = pyautogui.size()

        self.lstSpeeds = self.AvarageFilter(self.lstSpeeds, scale, totalXcount)
        avarageSpeed = np.average(self.lstSpeeds) / self.ratio
        maximumSpeed = np.max(self.lstSpeeds) / self.ratio
        v_x = np.linspace(0.0, totalXcount, np.size(self.lstSpeeds))
        v_y = np.divide(self.lstSpeeds, self.ratio)
        v_x_new = np.linspace(0, totalXcount, totalXcount * 100 + 1)

        speedGraph = self.drawGraph(v_x, v_y, v_x_new, avarageSpeed, "Velocity (mm / s)", "VelocityGraph")

        self.lstCounts = self.AvarageFilter(self.lstCounts, 100, totalXcount)
        self.lstCounts = np.array(self.lstCounts).astype('uint8')
        avarageCount = int(np.average(self.lstCounts))

        c_x = np.linspace(0.0, totalXcount, np.size(self.lstCounts))
        c_y = self.lstCounts
        c_x_new = np.linspace(0, totalXcount, totalXcount * 100 + 1)

        countGraph = self.drawGraph(c_x, c_y, c_x_new, avarageCount, "Count at Frame", "CountGraph ByFrame")

        graphs = np.hstack((speedGraph, countGraph))
        g_h = int(2 * s_height / 3)
        g_shade = np.shape(graphs)
        scale = g_shade[1] / s_width
        graphs = cv2.resize(graphs, (0, 0), None, scale, scale)

        width = int(graphs.shape[1])
        height = int(graphs.shape[0] / 2)
        blank_image = np.zeros((height, width, 3), np.uint8)
        blank_image = np.vstack((blank_image, graphs))

        onceNumber = self.roiWidth / avarageSpeed
        totalCount = np.sum(self.lstCounts) / onceNumber
        shape = np.shape(blank_image)
        point = int(shape[1] / 2) - 200
        draw_str(blank_image, (100 + point, 20), "----- Result -----")
        text = 'Average Speed : {:.6f} mm /s'.format(avarageSpeed)
        draw_str(blank_image, (point, 50), text)

        text = 'Maximum Speed : {:.6f} mm /s'.format(maximumSpeed)
        draw_str(blank_image, (point, 70), text)

        self.count_one_sec_list = np.array(self.count_one_sec_list)
        nan_arr = np.isnan(self.count_one_sec_list)
        non_nan_arr = ~nan_arr
        self.count_one_sec_list = self.count_one_sec_list[non_nan_arr]

        text = 'Average Count / s: {:.0f}'.format(
            int(np.average(self.count_one_sec_list))) + f'   |   Average Count / frame: {avarageCount}'
        draw_str(blank_image, (point, 90), text)

        text = 'Total Count : {:.0f}'.format(totalCount)
        draw_str(blank_image, (point, 110), text)
        #print(self.count_one_sec_list)
        #print(maximumSpeed)
        #print(int(np.average(self.count_one_sec_list)))
        p = os.path.join(self.path, self.filename + '.jpg')
        # print(p)
        cv2.imwrite(p, blank_image)
        # cv2.destroyWindow('ROI Detect')
        cv2.imshow('Base Frame', blank_image)
        cv2.resizeWindow('Base Frame', shape[1], shape[0])

        # ch = cv2.waitKey(0)
        plt.close('all')
        cv2.destroyAllWindows()
        # if ch == 27:
        #     cv2.destroyAllWindows()


def selFunc(val, canvas):
    global is_selected
    is_selected = val

    graph_path = os.path.join(os.getcwd(), 'Graphs')
    try:
        os.mkdir(os.path.join(graph_path))
    except Exception as e:
        print(e)
    dirs = os.listdir(videos_path)
    for d in dirs:
        if d == 'Medaka':
            type = 1
        else:
            type = 0
        video_files = glob.glob(os.path.join(videos_path, d, '*.mp4'))
        print("This", video_files)
        for file in video_files:
            global escape_pressed
            escape_pressed = False

            # print(file.split('\\'))
            print('Video selected is: {}'.format(file.split('\\')[1]))
            # print('Video selected is: {}'.format(file.split('/')[-1]))
            video_src = file
            try:
                static_ROI = App(canvas, 0.5, video_src, 50, graph_path, 200, type, is_selected, False).run()
            except Exception as e:
                print(e)
    # root.destroy()


def button_select_videos_folder_click():
    #print('button_select_videos_folder_click')
    global videos_path
    global var_label_selected_videos_folder

    videos_path = filedialog.askdirectory(initialdir=os.getcwd(), title='Select Folder')
    var_label_selected_videos_folder.set("videos folder : " + videos_path)

def show_results():
    path = os.path.realpath("Graphs")
    os.startfile(path)


def update():
    root.after(delay, update)


def on_escape_press(event):
    global escape_pressed
    escape_pressed = True
    #print('You pressed %s\n' % (event.char,))


escape_pressed = False
root = tk.Tk()
root.wm_title("Process Videos")
root.geometry("1000x650+100+10")
root.resizable(0, 0)

# Create a canvas that can fit the above video source size
canvas = tkinter.Canvas(root, width=650, height=500, bd=1, relief='ridge')
canvas.pack()
canvas.create_text(300, 200, fill="black", font="Times 24 bold",
                   text="Preview of Video")

file_header = PIL.Image.open("header.PNG")
img_header = PIL.ImageTk.PhotoImage(file_header)
label_header = tkinter.Label(root, image=img_header, height=30, width=700)

file_footer = PIL.Image.open("footer.PNG")
img_footer = PIL.ImageTk.PhotoImage(file_footer)
label_footer = tkinter.Label(root, image=img_footer, height=30, width=700)

file_fish = PIL.Image.open("fish.jpg")
img_fish = file_fish.resize((100, 100), PIL.Image.ANTIALIAS)
img_fish = PIL.ImageTk.PhotoImage(img_fish)
label_fish = tkinter.Label(root, image=img_fish, height=100, width=100)

file_opencv = PIL.Image.open("opencv.png")
img_opencv = file_opencv.resize((100, 100), PIL.Image.ANTIALIAS)
img_opencv = PIL.ImageTk.PhotoImage(img_opencv)
label_opencv = tkinter.Label(root, image=img_opencv, height=100, width=100)

var_label_selected_videos_folder = tkinter.StringVar()
label_selected_videos_folder = tkinter.Label(root, textvariable=var_label_selected_videos_folder, width=100)
var_label_selected_videos_folder.set("videos folder : ")

tkinter.Label(root, text="Press Escape to Process Next Video", width=100).place(x=120, y=580)

button_select_videos_folder = tkinter.Button(root, text="Select Videos Folder",
                                             command=button_select_videos_folder_click, bg='#808080', fg='#FFFFFF',
                                             height=5, width=20)
button_roi_automatic = tk.Button(root, text="Select ROI Automatic", command=lambda: selFunc(True, canvas), bg='#808080',
                                 fg='#FFFFFF', height=5, width=20)
button_roi_manual = tk.Button(root, text="Select ROI Manually", command=lambda: selFunc(False, canvas), bg='#808080',
                              fg='#FFFFFF', height=5, width=20)
button_results = tk.Button(root, text="Results", command=show_results, bg='#808080',
                              fg='#FFFFFF', height=5, width=20)

label_header.place(x=120, y=5)
label_footer.place(x=140, y=600)
label_fish.place(x=30, y=50)
label_opencv.place(x=30, y=200)
canvas.place(x=150, y=50)
button_select_videos_folder.place(x=820, y=50)
button_roi_automatic.place(x=820, y=150)
button_roi_manual.place(x=820, y=250)
button_results.place(x=820, y=350)
label_selected_videos_folder.place(x=100, y=560)

# img = PIL.ImageTk.PhotoImage(PIL.Image.open("opencv.png"))
# canvas.create_image(0, 0, anchor=tkinter.NW, image=img)

delay = 15
update()
root.bind('<Escape>', on_escape_press)
root.mainloop()
