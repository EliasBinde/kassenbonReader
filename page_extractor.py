import cv2
import numpy as np
from skimage.filters import threshold_otsu
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from itertools import combinations
from collections import defaultdict

from hough_line_corner_detector import HoughLineCornerDetector
from processors import Resizer, OtsuThresholder, FastDenoiser


class PageExtractor:
    def __init__(self, preprocessors, corner_detector, output_process = False):
        assert isinstance(preprocessors, list), "No processor list given"
        self._preprocessors = preprocessors
        self._corner_detector = corner_detector
        self.output_process = output_process


    def __call__(self, image_path):
        self._image = cv2.imread(image_path)

        self._processed = self._image
        for preprocessor in self._preprocessors:
            self._processed = preprocessor(self._processed)

        self._intersections = self._corner_detector(self._processed)

        return self._extract_page()


    def _extract_page(self):
        # obtain a consistent order of the points and unpack them
        # individually
        pts = np.array([
            (x, y)
            for intersection in self._intersections
            for x, y in intersection
        ])
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

    
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],                         
            [maxWidth - 1, 0],              
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]],            
            dtype = "float32"
        )

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(self._processed, M, (maxWidth, maxHeight))

        if self.output_process: cv2.imwrite('output/deskewed.jpg', warped)

        return warped

    
    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype = "float32")
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect





page_extractor = PageExtractor(
    preprocessors = [
        Resizer(height = 1280, output_process = True), 
        FastDenoiser(strength = 9, output_process = True),
        OtsuThresholder(output_process = True)
    ],
    corner_detector = HoughLineCornerDetector(
        rho_acc = 1,
        theta_acc = 180,
        thresh = 100,
        output_process = True
    )
)

def crop_page(image_path):
    page = page_extractor(image_path)
    return page

