__author__ = 'mearns'

import math
import numpy as np

def distance(a, b):
    '''Pythagoras: finds the distance between two points (a1, a2), (b1, b2)'''
    deltax2 = (b[0]-a[0])**2
    deltay2 = (b[1]-a[1])**2
    ab = math.sqrt(deltax2 + deltay2)
    return ab


def angleABC(a, b, c):
    '''Cosine rule: finds the angle at point b given three points (a1, a2), (b1, b2), (c1, c2)'''
    ab = distance(a, b)
    ac = distance(a, c)
    bc = distance(b, c)
    abc = math.acos(((bc**2)+(ab**2)-(ac**2))/(2*bc*ab))
    return abc


def gradient(a, b):
    '''finds the gradient between two points (a1, a2), (b1, b2)'''
    if a[0] != b[0]:
        m = (b[1]-a[1])/(b[0]-a[0])
    else:
        m = 1000
    return m


def invgrad(m):
    '''calculates perpendicular gradient'''
    if m != 0:
        inv = -1/m
    else:
        inv = 1000
    return inv


def yintercept(m, a):
    '''returns the y-intercept of line with gradient m through point (a1, a2)'''
    c = a[1] - (m * a[0])
    return c


def line(a, b):
    '''returns the gradient and y-intercept of line between through (a1, a2), (b1, b2)'''
    m = gradient (a, b)
    c = yintercept(m, a)
    return m, c


def lineintersect((m1, c1), (m2, c2)):
    '''finds the point of intersection between two lines y = (m1 * x) + c1, y = (m2 * x) + c2'''
    if m1 != m2:
        x = (c2 - c1)/(m1 - m2)
        y = (m1 * x) + c1
        intersect = (x, y)
        return intersect


def point2line(a, b, c):
    '''finds shortest distance between point c and line ab'''
    m1, y1 = line(a, b)
    m2 = invgrad(m1)
    y2 = yintercept(m2, c)
    d = lineintersect((m1, y1), (m2, y2))
    answer = distance(c, d)
    return answer


def angleAB(a, b):
    dx, dy = vector(a, b)
    angle = math.atan2(dy, dx)
    if angle < 0:
        angle += (2 * math.pi)
    return angle


def vector(a, b):
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    return (dx, dy)


def findClockwiseAngle(angle1, angle2, unit='radians'):
    '''finds the clockwise angle from angle1 to angle2'''
    if unit == 'radians':
        a1, a2 = (2 * math.pi + angle1) % (2 * math.pi), (2 * math.pi + angle2) % (2 * math.pi)
        difference = a2 - a1
        if difference < 0:
            difference = difference + 2 * math.pi
        return difference
    elif unit == 'degrees':
        a1, a2 = (360 + angle1) % 360, (360 + angle2) % 360
        difference = a2 - a1
        if difference < 0:
            difference = difference + 360
        return difference


def findMidpoint(*points):
    # find the midpoint, returns tuple
    n = len(points)
    xs, ys = zip(*points)
    x = sum(xs) * (1.0/n)
    y = sum(ys) * (1.0/n)
    return (x, y)


def angle2vector(rad):
    theta = rad % (2 * math.pi)
    v = np.array([1, math.tan(theta)])
    v /= np.linalg.norm(v)
    if math.pi / 2 < theta <= 3 * math.pi / 2:
        v *= -1
    return v

def dt_slope_2p(a,b):
    '''Calculate the slope between two points'''
    # Question. if I need to determine the abs value?
    # Question. temporarily can's solve the question when things are straight...
    return (a[1]-b[1])/(a[0]-b[0])

def dt_average(iterable):
    '''Take an interable like a list and returns its average'''
    return sum(iterable)/len(iterable)
