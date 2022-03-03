import cv2
from cv2 import imshow
import numpy as np
import matplotlib.pyplot as plt

#step2
def canny_image(image):                                 #function to get the gradient of the sharp edges in the image
    black = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)      #Convert the RGB image into black and white or grayscale single channel
    blur_image = cv2.GaussianBlur(black, (5,5), 0)      #Reduce noise in our image
    canny = cv2.Canny(blur_image, 50, 150)              #Get gradient of our blur image and apply Canny function
    return canny

#step3
def area_of_focus(image):
    height = image.shape[0]                              # i get the height of my image to be 0
    triangle = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])                                                #create an array of x and y axis, i put my traingle dimension into the array
    black_mask = np.zeros_like(image)                     #I convert my image into completely black mask so that i can insert my area of focus (triangle) 
    cv2.fillPoly(black_mask, triangle, 255)               #Put the triangle inside the black mask and giving triangle of color white 255
    masked_image = cv2.bitwise_and(image, black_mask)
    return masked_image

#step4
def display_line(image, lines):                          #I display the line i just detected on a black mask
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)                 #convert line into 1-d array of 4elements
            cv2.line(line_image, (x1,y1),(x2,y2), (0,250,0), 8)         #draw the line with green color and thickness of 10
    return line_image

#step5                                                  #i get my exact value for x1,x2,y1,y2
def coodinate(image, line_parameter):
    slope, intercept = line_parameter
    y1 = image.shape[0]                                  #i get the bottom height of my image
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def area_of_interest(image, lines):
    left_lines = []
    right_lines  =[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)                      #reshape the line in 1-D of 4 elements
        my_values = np.polyfit((x1,x2), (y1,y2), 1)        #calculate my slope and intercept in each line using polyFit function
        slope = my_values[0]
        intercept = my_values[1]
        #print(slope, 'slope')
        #print(intercept, 'intercept')
        if slope < 0:
            left_lines.append((slope, intercept))          #distinguish between left and right lines and assign them separately
        else:
            right_lines.append((slope, intercept))
    
    #print(average_left_line)
    #print(average_right_line)
    average_left_line = np.average(left_lines, axis=0)     #I get the average of slope and intercept
    average_right_line = np.average(right_lines, axis=0)
    
    
    final_left_line = coodinate(image, average_left_line)
    final_right_line = coodinate(image, average_right_line)
    return np.array([final_left_line, final_right_line])

video_capture = cv2.VideoCapture('road_video.mp4')
while(video_capture.isOpened()):
    _, my_video_frame = video_capture.read()
    canned = canny_image(my_video_frame)
    cropped_image = area_of_focus(canned)

    #Algorith to detect a line image from our cropped image
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=3)
    total_line_average = area_of_interest(my_video_frame, lines)

    #passing our cropped image as the first parameter, size of box in pixel, angle, empty array, minimum length of line and max gap between each line
    new_line_image = display_line(my_video_frame, total_line_average)

    #this command put our new detected line  image into the original colored image with the color image having weight of 0.7 and line image having weight of 1
    combine_images = cv2.addWeighted(my_video_frame, 0.7, new_line_image, 1, 1) 
    cv2.imshow('result', combine_images) 
    cv2.waitKey(1) 
    cv2.destroyAllWindows()

#plt.imshow(canny)
#plt.show()
