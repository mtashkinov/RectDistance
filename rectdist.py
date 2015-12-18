import cv2
import numpy as np
import math

def find_thresholds_theta(lines):
    theta1 = lines[0][0][1]
    theta2 = lines[0][0][1]
    max_diff = 0.0
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            if (abs(lines[i][0][1] - lines[j][0][1]) < abs(abs(lines[i][0][1] - lines[j][0][1]) - math.pi)):
                cur_diff = abs(lines[i][0][1] - lines[j][0][1])
            else:
                cur_diff = abs(abs(lines[i][0][1] - lines[j][0][1]) - math.pi)
            if cur_diff > max_diff:
                max_diff = cur_diff
                theta1 = lines[i][0][1]
                theta2 = lines[j][0][1]
    thr1 = (theta1 + theta2) / 2
    if thr1 > math.pi / 2:
        thr2 = thr1 - math.pi / 2
    else:
        thr2 = thr1
        thr1 = thr2 + math.pi / 2
    return thr1, thr2


def classify_theta(lines, (threshold1, threshold2)):
    first = []
    second = []
    for x in lines:
        if (x[0][1] < threshold1) & (x[0][1] > threshold2):
            first.append(x[0])
        else:
            second.append(x[0])

    return first, second


def find_cross_point(line1, line2):
    b1 = np.cos(line1[1])
    a1 = np.sin(line1[1])
    x1 = b1*line1[0]
    y1 = a1*line1[0]
    b2 = np.cos(line2[1])
    a2 = np.sin(line2[1])
    x2 = b2*line2[0]
    y2 = a2*line2[0]
    t2 = (a1 * (y1 - y2) + b1 * (x1 - x2)) / (a1 * b2 - a2 * b1)
    return x2 - t2 * a2, y2 + t2 * b2


def find_cross_points(line, lines):
    cross_points = []
    for x in lines:
        cross_points.append(find_cross_point(line, x))
    return cross_points


def find_threshold_cross_points(cross_points):
    min = cross_points[0][0]
    max = cross_points[0][0]
    for x in cross_points:
        if x[0] < min:
            min = x[0]
        elif x[0] > max:
            max = x[0]
    return (max + min) / 2


def classify_cross_points(lines, cross_points, threshold):
    first = []
    second = []
    for i in range(len(cross_points)):
        if cross_points[i][0] < threshold:
            first.append(lines[i])
        else:
            second.append(lines[i])

    return first, second


def classify_lines(lines):
    class_theta = classify_theta(lines, find_thresholds_theta(lines))
    cross_points = find_cross_points(class_theta[1][0], class_theta[0])
    first, third = classify_cross_points(class_theta[0], cross_points, find_threshold_cross_points(cross_points))
    cross_points = find_cross_points(class_theta[0][0], class_theta[1])
    second, fourth = classify_cross_points(class_theta[1], cross_points, find_threshold_cross_points(cross_points))
    return first, second, third, fourth


def check_theta(lines):
    min = lines[0][1]
    max = lines[0][1]
    for x in lines:
        if x[1] > max:
            max = x[1]
        elif x[1] < min:
            min = x[1]
    return max - min > math.pi / 2



def filter_lines(lines):
    average_the = 0.0
    average_rho = 0.0
    p = check_theta(lines)
    for x in lines:
        cur_the = x[1]
        if p & (x[1] > math.pi / 2):
            cur_the -= math.pi
        average_the += cur_the
        average_rho += x[0]
    average_the /= len(lines)
    average_rho /= len(lines)

    rho = lines[0][0]
    theta = lines[0][1]
    dist = abs(lines[0][1] - average_the)
    for x in lines:
        cur_dist = abs(x[1] - average_the)
        if cur_dist < dist:
            theta = x[1]
            rho = x[0]
            dist = cur_dist
    return rho, theta


def draw_line(img, (rho, theta), color):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))

    cv2.line(img, (x1,y1), (x2,y2), color, 5)


def find_red_contour(img):
    img = cv2.GaussianBlur(img, (5, 5), 5)
    mask = cv2.inRange(img, (0, 0, 230), (200, 200, 255))
    im2, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = contours[0]
    for contour in contours:
        if cv2.contourArea(contour) > cv2.contourArea(max_contour):
            max_contour = contour
    return max_contour

def detect_rect(img):
    empty = np.zeros((img.shape[0], img.shape[1]))
    contour = find_red_contour(img)
    cv2.drawContours(empty, [contour], 0, (255, 255, 255), 2)
    im2 = np.array(255 * empty, dtype=np.uint8)
    lines = cv2.HoughLines(im2,1,np.pi/180, 100)
    sides = classify_lines(lines)
    return sides, contour


def count_sqr_distance(point1, point2):
    return (point1[0] - point2[0]) * (point1[0] - point2[0]) + (point1[1] - point2[1]) * (point1[1] - point2[1])


def find_farthest_point(contour, point):
    max_dist = 0.0
    vertex = point
    for x in contour:
        cur_dist = count_sqr_distance(point, x[0])
        if cur_dist > max_dist:
            vertex = x[0]
            max_dist = cur_dist
    return vertex


def count_sqr_distance_point_line(a, b, c, point):
    return abs(a * point[0] + b * point[1] + c + 0.0) / (a * a +  b * b)


def find_farthest_point_line(contour, point1, point2):
    a = point2[1] - point1[1]
    b = point1[0] - point2[0]
    c = point1[1] * point2[0] - point1[0] * point2[1]
    max_dist = 0.0
    vertex = point1
    for x in contour:
        cur_dist = count_sqr_distance_point_line(a, b, c, x[0])
        if cur_dist > max_dist:
            max_dist = cur_dist
            vertex = x[0]
    return vertex


def detect_rect1(img):
    img = cv2.GaussianBlur(img, (5, 5), 5)
    mask = cv2.inRange(img, (0, 0, 230), (30, 30, 255))

    im2, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = contours[0]
    for contour in contours:
        if cv2.contourArea(contour) > cv2.contourArea(max_contour):
            max_contour = contour

    #vertexes = []
    #vertexes.append(find_farthest_point(max_contour, max_contour[0][0]))
    #vertexes.append(find_farthest_point(max_contour, vertexes[0]))
    #vertexes.append(find_farthest_point_line(max_contour, vertexes[0], vertexes[1]))
    #vertexes.append(find_farthest_point(max_contour, vertexes[2]))

    #return vertexes


img = cv2.imread('ex\\1.jpg')
res = detect_rect(img)
empty = np.zeros((img.shape[0], img.shape[1], 3))
sides, contour = detect_rect(img)
draw_line(empty, filter_lines(sides[0]), (255, 0, 255))
draw_line(empty, filter_lines(sides[1]), (255, 0, 255))
draw_line(empty, filter_lines(sides[2]), (255, 0, 255))
draw_line(empty, filter_lines(sides[3]), (255, 0, 255))
cv2.drawContours(empty, [contour], 0, (0, 255, 0), 5)

empty = cv2.resize(empty, (int(empty.shape[1] * (512.0 / empty.shape[0])), 512))
cv2.imshow('img', empty)
cv2.waitKey(0)
