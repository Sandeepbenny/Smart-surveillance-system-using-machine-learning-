from django.shortcuts import render
import cv2
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
from skimage.metrics import structural_similarity
import winsound
import time
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import FileResponse
import os
from django.core.mail import EmailMessage
from django.conf import settings


# chatgpt working

import cv2
from skimage.metrics import structural_similarity
import winsound
from django.core.mail import EmailMessage
from django.conf import settings

from django.core.mail import EmailMessage
from django.conf import settings


def spot_diff(frame1, frame2):
    # your existing spot_diff code here
    
    frame1 = frame1[1]
    frame2 = frame2[1]

    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    g1 = cv2.blur(g1, (2,2))
    g2 = cv2.blur(g2, (2,2))

    (score, diff) = structural_similarity(g2, g1, full=True)

    print("Image similarity", score)

    diff = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY_INV)[1]

    contors = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contors = [c for c in contors if cv2.contourArea(c) > 50]

    if len(contors):
        for c in contors:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame1, (x,y), (x+w, y+h), (0,255,0), 2)
        winsound.Beep(1000, 500)
        
        cv2.imshow("diff", thresh)
        cv2.imshow("win1", frame1)
        cv2.imwrite("stolen/stolen.jpg", frame1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # send an email with the stolen object image
        subject = 'Object Stolen'
        message = 'The following object has been stolen:'
        from_email = 'sanduchackonal@gmail.com'
        recipient_list = ['sandeepchackonal@gmail.com', 'sanduchackonal@gmail.com']  # add your recipient email addresses here
        image_path = 'stolen/stolen.jpg'  # assuming the image is saved in 'stolen' folder with filename 'stolen.jpg'

        # create EmailMessage object and attach the image file to it
        email = EmailMessage(subject, message, from_email, recipient_list)
        email.content_subtype = "html"
        email.attach_file(image_path)

        # send the email
        email.send()
        
    else:
        print("nothing stolen")
        return 0

    return 1



import cv2
import time
from django.core.mail import EmailMessage
from django.conf import settings
from django.core.mail import send_mail

from django.core.mail import EmailMessage

def find_motion():
    motion_detected = False
    is_start_done = False

    cap = cv2.VideoCapture(0)

    check = []

    print("waiting for 2 seconds")
    time.sleep(2)
    frame1 = cap.read()

    _, frm1 = cap.read()
    frm1 = cv2.cvtColor(frm1, cv2.COLOR_BGR2GRAY)

    while True:
        _, frm2 = cap.read()
        frm2 = cv2.cvtColor(frm2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(frm1, frm2)

        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        contors = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # look at it
        contors = [c for c in contors if cv2.contourArea(c) > 25]

        if len(contors) > 5:
            cv2.putText(thresh, "motion detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            motion_detected = True
            is_start_done = False

        elif motion_detected and len(contors) < 3:
            if (is_start_done) == False:
                start = time.time()
                is_start_done = True
                end = time.time()

            end = time.time()

            print(end-start)
            if (end - start) > 4:
                frame2 = cap.read()
                cap.release()
                cv2.destroyAllWindows()
                x = spot_diff(frame1, frame2)
                if x == 0:
                    print("running again")
                    return

                else:
                    print("found motion")
                    # Create the email message object
                    subject = 'Motion Detected'
                    from_email = 'sanduchackonal@gmail.com'
                    recipient_list = ['sandeepchackonal@gmail.com']
                    message = 'Motion has been detected.'
                    email = EmailMessage(subject, message, from_email, recipient_list)
                    with open('stolen\stolen.jpg', 'rb') as f:
                        email.attach('image.jpg', f.read(), 'image/jpeg')
                    # Send the email
                    email.send()
                    return

        else:
            cv2.putText(thresh, "no motion detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        cv2.imshow("Window", thresh)

        _, frm1 = cap.read()
        frm1 = cv2.cvtColor(frm1, cv2.COLOR_BGR2GRAY)

        if cv2.waitKey(1) == 27:
            break

    return



    

def motion_detection_view(request):
    find_motion()
    return render(request, 'motion_detected.html')


# cgpt
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    return faces, gray


def process_frame(frame, name, ids, count):
    faces, gray = detect_faces(frame)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        roi = gray[y:y+h, x:x+w]
        cv2.imwrite(f"persons/{name}-{count}-{ids}.jpg", roi)
        count += 1
        cv2.putText(frame, f"{count}", (20,20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 3)
        cv2.imshow("new", roi)
    return frame, count


def train(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        ids = request.POST.get('id')
        count = 1
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame, count = process_frame(frame, name, ids, count)
            cv2.imshow("identify", frame)
            if cv2.waitKey(1) == 27 or count > 350:
                cv2.destroyAllWindows()
                cap.release()
                dataset = 'persons'
                paths = [os.path.join(dataset, im) for im in os.listdir(dataset)]
                faces = []
                ids = []
                labels = []
                for path in paths:
                    labels.append(path.split('/')[-1].split('-')[0])
                    ids.append(int(path.split('/')[-1].split('-')[2].split('.')[0]))
                    faces.append(cv2.imread(path, 0))
                recog = cv2.face.LBPHFaceRecognizer_create()
                recog.train(faces, np.array(ids))
                recog.save('model.yml')
                break
    return render(request, 'train.html')


import os

def save_unknown_face(roi, count):
    unknown_folder = "unknown"
    if not os.path.exists(unknown_folder):
        os.makedirs(unknown_folder)
    cv2.imwrite(f"{unknown_folder}/unknown-{count}.jpg", roi)

# chat gpt for saving faces
def identify_face(request):
    filename = "haarcascade_frontalface_default.xml"
    recog = cv2.face.LBPHFaceRecognizer_create()
    recog.read('model.yml')
    cascade = cv2.CascadeClassifier(filename)
    cap = cv2.VideoCapture(0)
    labelslist = {}
    paths = [os.path.join("persons", im) for im in os.listdir("persons")]
    for path in paths:
        labelslist[path.split('/')[-1].split('-')[2].split('.')[0]] = path.split('/')[-1].split('-')[0]
    
    count = 1  # Counter for unknown faces
    
    while True:
        _, frm = cap.read()
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 5)
    
        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (100, 100))
            label = recog.predict(roi)
    
            if label[1] < 70:
                cv2.putText(frm, f"{labelslist[str(label[0])]} + {int(label[1])}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(frm, "unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                save_unknown_face(roi, count)  # Save the unknown face
                count += 1
    
        cv2.imshow("identify", frm)
    
        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            break
    
    return render(request, 'identify.html')




def record(request):
    cap = cv2.VideoCapture(0)

    # Specify the video codec and other parameters
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(f'recordings/{datetime.now().strftime("%H-%M-%S")}.avi', fourcc,20.0,(640,480))

    # Loop to read frames from the camera and write them to the output video file
    while True:
        _, frame = cap.read()

        # Add timestamp to the frame
        cv2.putText(frame, f'{datetime.now().strftime("%D-%H-%M-%S")}', (50,50), cv2.FONT_HERSHEY_COMPLEX,
                    0.6, (255,255,255), 2)

        out.write(frame)

        # Display the frame in a window
        cv2.imshow("esc. to stop", frame)

        # Check for the ESC key to stop the recording
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break 

    return render(request, 'record.html')
# chatgpt
import cv2
import pygame

def motion_detection(request):
    # initialize variables
    donel = False
    doner = False
    x1, y1, x2, y2 = 0, 0, 0, 0
    
    # initialize sound
    pygame.mixer.init()
    sound = pygame.mixer.Sound('alarm.wav')
    
    def select(event, x, y, flags, param):
        nonlocal x1, x2, y1, y2, donel, doner
        if event == cv2.EVENT_LBUTTONDOWN:
            x1, y1 = x, y
            donel = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            x2, y2 = x, y
            doner = True

    def rect_noise():
        nonlocal x1, x2, y1, y2, donel, doner
        cap = cv2.VideoCapture(0)

        cv2.namedWindow("select_region")
        cv2.setMouseCallback("select_region", select)

        while True:
            _, frame = cap.read()

            cv2.imshow("select_region", frame)

            if cv2.waitKey(1) == 27 or doner:
                cv2.destroyAllWindows()
                break

        while True:
            _, frame1 = cap.read()
            _, frame2 = cap.read()

            frame1only = frame1[y1:y2, x1:x2]
            frame2only = frame2[y1:y2, x1:x2]

            diff = cv2.absdiff(frame2only, frame1only)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            diff = cv2.blur(diff, (5, 5))
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            contr, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contr) > 0:
                max_cnt = max(contr, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(max_cnt)
                cv2.rectangle(frame1, (x + x1, y + y1), (x + w + x1, y + h + y1), (0, 255, 0), 2)
                cv2.putText(frame1, "MOTION", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                sound.play() # play sound when motion is detected

            else:
                cv2.putText(frame1, "NO-MOTION", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.imshow("esc. to exit", frame1)

            if cv2.waitKey(1) == 27:
                cap.release()
                cv2.destroyAllWindows()
                break

        pygame.mixer.quit()
        cap.release()
        cv2.destroyAllWindows()

    rect_noise()

    return render(request, 'motion_detection.html')
import os
from datetime import datetime
import cv2
from django.shortcuts import render

# in out
def in_out(request):
    render(request, 'in_out.html')
    cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    message_time = None
    x = 300
    counter = 0

    while True:
        _, frame1 = cap.read()
        frame1 = cv2.flip(frame1, 1)
        _, frame2 = cap.read()
        frame2 = cv2.flip(frame2, 1)

        diff = cv2.absdiff(frame2, frame1)
        diff = cv2.blur(diff, (5, 5))
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, threshd = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        contr, _ = cv2.findContours(
            threshd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contr) > 0:
            max_cnt = max(contr, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_cnt)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "MOTION", (10, 80),
                        font, 2, (0, 255, 0), 2)

            if x > 500:
                direction = "left"
            elif x < 200:
                direction = "right"
            else:
                direction = ""

            if direction:
                counter += 1
                cv2.putText(frame1, f"Person moved from {direction} to oppposite direction",
                            (10, 40), font, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                x = 300
                path = f"visitors/{direction}/"
                filename = f"{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{counter}.jpg"
                fullpath = os.path.join(path, filename)
                cv2.imwrite(fullpath, frame1)
                message_time = int(datetime.now().timestamp())

        if message_time and (datetime.now() - datetime.fromtimestamp(message_time)).total_seconds() < 2:
            cv2.putText(frame1, "Saved image", (10, 120),
                        font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.namedWindow("Smart CCTV", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Smart CCTV", (700, 500))
        cv2.imshow("Smart CCTV", frame1)

        k = cv2.waitKey(1)

        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
    return render(request, 'in_out.html')

class MotionDetector:
    def __init__(self, video_source=0, alarm_sound_file="alarm.wav"):
        self.cap = cv2.VideoCapture(video_source)
        self.alarm_sound_file = alarm_sound_file
        self.motion_detected = False
        self.sound_playing = False # keep track of whether sound is currently playing

    def detect_motion(self):
        while True:
            _, frame1 = self.cap.read()
            _, frame2 = self.cap.read()

            diff = cv2.absdiff(frame2, frame1)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            diff = cv2.blur(diff, (5,5))
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            contr, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(contr) > 0:
                max_cnt = max(contr, key=cv2.contourArea)
                x,y,w,h = cv2.boundingRect(max_cnt)
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame1, "MOTION", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)

                if not self.motion_detected:
                    # start a new thread to play the alarm sound
                    threading.Thread(target=self.play_sound).start()
                    self.motion_detected = True
            else:
                cv2.putText(frame1, "NO-MOTION", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                self.motion_detected = False

            cv2.imshow("Motion Detector", frame1)

            # check if sound has finished playing
            if self.sound_playing and not sd.get_stream().active:
                self.sound_playing = False

            if cv2.waitKey(1) == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def play_sound(self):
        # load the sound file
        data, fs = sf.read(self.alarm_sound_file, dtype='float32')
        # play the sound
        sd.play(data, fs)
        # set the sound_playing flag
        self.sound_playing = True
        # wait for the sound to finish playing
        sd.wait()

@csrf_exempt
def motion_detector_view(request):
    if request.method == 'POST':
        # get the file uploaded by the client
        alarm_file = request.FILES.get('alarm.wav', None)
        # save the file to a temporary location
        if alarm_file:
            with open('alarm.wav', 'wb') as f:
                for chunk in alarm_file.chunks():
                    f.write(chunk)

        # create a motion detector instance and detect motion
        md = MotionDetector(alarm_sound_file='alarm.wav')
        md.detect_motion()

    return render(request, 'alarm.html')
from django.http import HttpResponse

def default(request):
    return HttpResponse("Hello, world!")

def download_executable(request):
    file_path = os.path.join(settings.STATIC_ROOT, 'main.exe')
    with open(file_path, 'rb') as file:
        response = HttpResponse(file.read(), content_type='application/octet-stream')
        response['Content-Disposition'] = 'attachment; filename="main.exe"'
        return response


def home(request):
     return render(request, 'home.html')
 
def features(request):
   
     return render(request,'features.html')  
 
def mainpage(request):
    return render(request,'mainpage.html')
 