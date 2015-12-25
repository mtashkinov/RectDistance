import cv2
import numpy as np
import math
from scipy.optimize import root
from scipy.optimize import fsolve


f = 2220
c = (940, 1251)


def find_thresholds_theta(lines):
    theta1 = lines[0][0][1]
    theta2 = lines[0][0][1]
    max_diff = 0.0
    for i in range(len(lines)):
        for j in range(i, len(lines)):
            if abs(lines[i][0][1] - lines[j][0][1]) < abs(abs(lines[i][0][1] - lines[j][0][1]) - math.pi):
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


def filter_contours(contours, areas, image_area):
    res = []
    for i in range(len(areas)):
        if areas[i] > image_area * 0.01:
            res.append(contours[i])
    return res
def find_red_contours(img):
    img = cv2.GaussianBlur(img, (11, 11), 11)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img, (160, 50, 50), (179, 255, 255))
    mask2 = cv2.inRange(img, (0, 50, 50), (10, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.dilate(mask, ())
    _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    fst_contour = contours[0]
    fst_area = cv2.contourArea(fst_contour)
    snd_area = 0
    snd_contour = []
    for contour in contours:
        cur_area = cv2.contourArea(contour)
        if cur_area > fst_area:
            snd_contour = fst_contour
            snd_area = fst_area
            fst_contour = contour
            fst_area = cur_area
        elif cur_area > snd_area:
            snd_contour = contour
            snd_area = cur_area
    res = [fst_contour, snd_contour]
    areas = [fst_area, snd_area]
    return filter_contours(res, areas, img.shape[0] * img.shape[1])


def detect_rects(img):
    res = []
    contours = find_red_contours(img)
    for contour in contours:
        empty = np.zeros((img.shape[0], img.shape[1]))
        cv2.drawContours(empty, [contour], 0, (255, 255, 255), 2)
        im2 = np.array(255 * empty, dtype=np.uint8)
        lines = cv2.HoughLines(im2,1,np.pi/180, 100)
        sides = classify_lines(lines)
        res.append((filter_lines(sides[0]), filter_lines(sides[1]), filter_lines(sides[2]), filter_lines(sides[3])))
    return res, contours


def sqr_dist(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def get_max_len(points):
    return max(sqr_dist(points[i], points[i + 1]) for i in range(-1, 3))


def fun(z):
    z_b = [[z[j]**2, - 2*z[j]*z[i], z[i]**2] for i in range(3) for j in range(i)]

    f = [sum(z_b[j][i] * b[j][i] for i in range(3)) - rect_size[j] for j in range(3)]

    return f


def find_distance(lines, size):
    global b
    global rect_size

    rect_size = size
    rect_points = [find_cross_point(lines[i], lines[i + 1]) for i in range(3)] + [find_cross_point(lines[3], lines[0])]

    a = [[(rect_points[i][j] - c[j]) / f for j in range(2)] for i in range(3)]
    b = [[a[j][0]**2 + a[j][1]**2 + 1, a[j][0]*a[i][0] + a[j][1]*a[i][1] + 1, a[i][0]**2 + a[i][1]**2 + 1]
     for i in range(3) for j in range(i)]

    estimate = math.sqrt(29.7**2 + 21.0**2) * f / math.sqrt(sum((rect_points[0][i] - rect_points[2][i])**2
                        for i in range(2)))
    sol = fsolve(fun, 3*[estimate])

    res_points = [(sol[i] * a[i][0], sol[i] * a[i][1], sol[i]) for i in range(3)]
    res = [(res_points[0][i] + res_points[2][i]) / 2 for i in range(3)]

    center = [int((rect_points[0][i] + rect_points[2][i]) / 2) for i in range(2)]
    pr = projection(res)

    return res, sqr_dist(center, pr), tuple(center)


def projection(point):
    return f * point[0] / point[2] + c[0], f * point[1] / point[2] + c[1]


b = []
rect_size = ()

video = cv2.VideoCapture('VID_20151225_195650.mp4')

flag, frame = video.read()

width = np.size(frame, 1)
height = np.size(frame, 0)

writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MJPG'),
    20, (width, height))

while flag:
    flag, frame = video.read()
    if flag == 0:
        break

    rows, cols, _ = frame.shape

    M = cv2.getRotationMatrix2D((cols/2, rows/2), -90, 1)
    frame = cv2.warpAffine(frame, M, (cols, rows))
    try:
        sides, contour = detect_rects(frame)
        #for i in range(len(sides)):
        for i in range(1):
            p2 = find_distance(sides[i], (29.7**2, 29.7**2 + 21.0**2, 21.0**2))
            p1 = find_distance(sides[i], (21.0**2, 29.7**2 + 21.0**2, 29.7**2))

            if p1[1] > p2[1]:
                p = p2
            else:
                p = p1

            dist = str(math.sqrt(sum(p[0][i]**2 for i in range(3))))
            cv2.putText(frame, dist, p[2], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 1, cv2.LINE_AA)
    except Exception:
        None

    writer.write(frame)

video.release()
cv2.destroyAllWindows()

cv2.waitKey(0)
