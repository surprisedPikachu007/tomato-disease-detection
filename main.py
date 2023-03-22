import datetime
import cv2
import requests

API_URL = "https://api-inference.huggingface.co/models/surprisedPikachu007/tomato-disease-detection_V3"
headers = {"Authorization": "Bearer <API_KEY>"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def predict(image):
    cv2.imwrite('test.jpg', image)
    result = query('test.jpg')
    return result
        
camera = cv2.VideoCapture(0)

while True:
    ret, image = camera.read()
    cv2.imshow("Image", image)
    
    result = predict(image)
    print('------------')
    print((datetime.datetime.now()).strftime("%H:%M:%S"))
    
    try:
        print(result[0]['label'])
    except:
        print()
    
    keyboard_input = cv2.waitKey(1)

    if keyboard_input == 27:
        break
    
camera.release()
cv2.destroyAllWindows()

    