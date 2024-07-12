import cv2
import numpy as np

video_path = r"C:\Users\Betty Go\Downloads\swing2.mp4"
at_first = 1

def mouse_callback(event, x, y, flags, img, param=True):
    global clicked_point, selected_area_name
    global pose
    global at_first

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        if param:
            if at_first :
                hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                clicked_hsv_color = hsv_image[clicked_point[1], clicked_point[0]]

                # 클릭한 픽셀과 유사한 색상 추출
                color_range = 50  # 색상 범위
                lower_bound = np.array([clicked_hsv_color[0] - color_range, 50, 50])
                upper_bound = np.array([clicked_hsv_color[0] + color_range, 255, 255])

                mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
                result = cv2.bitwise_and(img, img, mask=mask)

                # 연속된 픽셀을 묶기
                gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 클릭한 픽셀이 포함된 무리 찾기
                clicked_cluster = None
                for contour in contours:
                    if cv2.pointPolygonTest(contour, clicked_point, False) >= 0:
                        clicked_cluster = contour
                        break

                if clicked_cluster is not None:

                    x, y, w, h = cv2.boundingRect(clicked_cluster) # 자르기
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    clicked_area = img[y:y + h, x:x + w]
                    saved_areas[selected_area_name] = {'coordinates': (x, y, w, h)}
        #선택한 영역 이름
        at_first=0
        clicked_positions = []
        text_positions = [(50, 100), (150, 100), (250, 100)]
        for i, (text_x, text_y) in enumerate(text_positions, start=1):
            if i == 1:
                cv2.putText(img, 'slide', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif i == 2:
                cv2.putText(img, 'swing', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif i == 3:
                cv2.putText(img, 'else', (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            clicked_positions.append((text_x, text_y))
        for i, (text_x, text_y) in enumerate(clicked_positions, start=1):
            if text_x < x < text_x + 100 and text_y - 30 < y < text_y + 10:
                if i==1 : selected_area_name='slide'
                elif i==2 : selected_area_name='swing'
                elif i==3 : selected_area_name = 'else'
                print("Mouse clicked at (x={}, y={}) - {}".format(x, y, selected_area_name))
                break
        else:
            print("Mouse clicked at (x={}, y={}) - Outside text area".format(x, y))


#초기 입력값
clicked_point = None
selected_area_name = ""
cnt=0
saved_areas = {}


cap = cv2.VideoCapture(video_path)

ret, first_img = cap.read()
cv2.namedWindow('First Image')


while True:

    first_img = cv2.resize(first_img, (400, 700))
    cv2.imshow('First Image', first_img)
    cv2.setMouseCallback('First Image', mouse_callback, first_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'): #창 닫기
        break
cv2.destroyAllWindows()



xy_list_list = []

while True:
    ret, image = cap.read()
    if not ret:
        break
    image = cv2.resize(image, (400, 700))


    cv2.imshow('Image', image)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('a'):  # 창 닫기
        break

cv2.destroyAllWindows()


# 저장된 영역과 좌표 출력
for area_name, area_info in saved_areas.items():
    print(f"영역명: {area_name}, 좌표: {area_info['coordinates']}")

