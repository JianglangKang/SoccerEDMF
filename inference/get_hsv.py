# 取色器
# http://www.jiniannet.com/Page/allcolor
# 将获取的四个点的rgb值填入下方的rgb，即可得到四个点的hsv上下限
import cv2
import numpy as np
# todo 获取球衣颜色
# rgb='#7A3F43,#83454F,#A06064,#CE8591'
# rgb='#E8E7ED,#E4E4DE,#B6BEB3,#B5B7B6'
# rgb='#6CB5DB,#54B5ED,#AACFDA,#9DBCCE'  #blue
rgb='#549BC7,#70A9CF,#7BA4B6,#BBD4EB,#91C2D2'
# rgb='#B9616F,#9E5F67,#8B5A5D,#8F4050'
# green
# rgb = '#dd343a,#eb2a35,#b46d7f,#b3222d'



# yellow
# rgb = '#dbf534,#dfe501,#ecef67,#e9ed4d'

rgb = rgb.split(',')

# 转换为BGR格式，并将16进制转换为10进制
bgr = [[int(r[5:7], 16), int(r[3:5], 16), int(r[1:3], 16)] for r in rgb]

# 转换为HSV格式
hsv = [list(cv2.cvtColor(np.uint8([[b]]), cv2.COLOR_BGR2HSV)[0][0]) for b in bgr]

hsv = np.array(hsv)
print('H:', min(hsv[:, 0]), max(hsv[:, 0]))
print('S:', min(hsv[:, 1]), max(hsv[:, 1]))
print('V:', min(hsv[:, 2]), max(hsv[:, 2]))
