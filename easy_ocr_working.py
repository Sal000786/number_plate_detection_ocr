import cv2
import easyocr
import imutils
import numpy as np
import matplotlib.pyplot as plt

img_path="F:\\Salman codes\\Open_CV_Course\\OCR(Optical_Character_Recognition)\\number_plate3.jpg"

image = cv2.imread(img_path)
dst = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
cv2.imshow("denoising",dst)
cv2.waitKey(0)

image_gray=cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)
blur = cv2.bilateralFilter(image_gray,9,75,75)
cv2.imshow("blur",blur)
cv2.waitKey(0)


# thres=cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
# edges=cv2.Canny(thres,30,200)
# cv2.imshow("edges",edges)
# cv2.waitKey(0)

reader = easyocr.Reader(['en'])
text = reader.readtext(blur)
print(text)
# if text[-1]>=60:
#     final_text=text[-2]
# print(final_text)

# for i in text:
#      for j in i:
#         if text[i][-1]>=0.6:

#             final_text=j[-2]
# print(fianl_text)

# font = cv2.FONT_HERSHEY_SIMPLEX
# res = cv2.putText(image, text=text[0][-2], fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
# res = cv2.rectangle(image, (text), (0,255,0),3)
# plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

# # # Find contours
# keypoints = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours = imutils.grab_contours(keypoints)
# # contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
# cv2.drawContours(image,contours,-1,(0,255,0),2)
# cv2.imshow("contours",image)
# cv2.waitKey(0)

# location = None
# for contour in contours:
#     approx = cv2.approxPolyDP(contour, 10, True)
#     if len(approx) == 4:
#         location = approx
#         break

# mask = np.zeros(image_gray.shape, np.uint8)
# new_image = cv2.drawContours(mask, [location], 0,255, -1)
# new_image = cv2.bitwise_and(image, image, mask=mask)

# (x,y) = np.where(mask==255)
# (x1, y1) = (np.min(x), np.min(y))
# (x2, y2) = (np.max(x), np.max(y))
# cropped_image = image_gray[x1:x2+1, y1:y2+1]

# cv2.imshow("cropped image",cropped_image)
# cv2.waitKey(0)

# area_list=[]
# for contour in contours:
#     area = cv2.contourArea(contour)
#     area_list.append(area)
#     # print(sorted(area,reverse=False))
 
# sorted_list=sorted(area_list,reverse=True)[:10]
# print(sorted_list)

# area_threshold = 400  # Adjust this value based on your specific image and use case

# Iterate through the contours and find the one corresponding to the number plate
# for contour in contours:
#     area = cv2.contourArea(contour)
#     if area > area_threshold:
#         # This contour may be the number plate
#         x, y, w, h = cv2.boundingRect(contour)
#         number_plate_region = image[y:y+h, x:x+w]

#         # Do further processing or analysis with the extracted number plate region
#         cv2.imshow('Number Plate', number_plate_region)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

# x,y,w,h = cv2.boundingRect(contours)
# cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
# cv2.imshow("bounding rec",image)
# cv2.waitKey(0)

# cv2.drawContours(image,contours,-1,(0,255,0),2)
# cv2.imshow("contours",image)
# cv2.waitKey(0)


# # Find potential number plate regions
# potential_regions = []
# for cnt in contours:
#     x, y, w, h = cv2.boundingRect(cnt)
#     aspect_ratio = float(w) / h
#     area = w * h
#     # Filter based on aspect ratio and area (adjust thresholds as needed)
#     if 2.5 < aspect_ratio < 4.0 and 5000 < area < 20000:
#         print(aspect_ratio,area)
#         potential_regions.append((x, y, w, h))
#         print("potentioal",potential_regions)



# print(potential_regions)
# print(len(potential_regions))
# Apply OCR to potential regions
# reader = easyocr.Reader(['en'])
# for x, y, w, h in potential_regions:
#     roi = image_gray[y:y+h, x:x+w]

# cv2.imshow("roi",roi)
# cv2.waitKey(0)

#     text = reader.readtext(roi, detail=0)

#     if text:
#         number = text[0][1]  # Extract the number
#         print("Detected number:", number)

#         # Draw rectangle around number plate region
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(image, number, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# # Display image with detected number (optional)
# cv2.imshow("Number Plate Detection", image)
# cv2.waitKey(0)
