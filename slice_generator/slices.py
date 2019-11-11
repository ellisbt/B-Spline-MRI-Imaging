"""
try to load and print a slice from MRcolorData
"""
import pymysql
import time
import numpy as np
from PIL import Image


def mri2d(x, y, color, filename):
    x = np.asarray(x)
    y = np.asarray(y)
    c = np.asarray(color)
    im = Image.new("RGB", (np.amax(x) + 1, np.amax(y) + 1))
    pix = im.load()
    for i in range(x.shape[0]):
        pix[int (x[i]), int (y[i])] = (int (c[i]), int (c[i]), int (c[i]))
    im.save(filename, "PNG")


'''a = input("Input slice #(0-143): ")
start = time.time()
conn = pymysql.connect(host='localhost', user='root', passwd='', db='mridata')
with conn.cursor() as cursor:
    cursor.execute("SELECT * FROM `pointdata` WHERE pointdata.splNum = " + str(a))
    result = np.array(cursor.fetchall())
conn.close()
print("it takes " + str(time.time()-start) +  " seconds")'''
# a = input("Input slice #(0-143):")
for k in range(144):
    dst = "splice" + str(k) + ".txt"
    result = np.genfromtxt(dst, delimiter=",", dtype=int)
    x = np.array(result[:, 1])
    y = np.array(result[:, 2])
    colorIntensity = np.array(result[:, 3])
    file_name = "test" + str(k) + ".png"
    mri2d(x, y, colorIntensity, file_name)