from django.shortcuts import render, redirect
import numpy as np
import cv2
import pickle
import os
from django.contrib import messages
from django.conf import settings as st
from PIL import Image
from django.contrib.auth.decorators import login_required
from .models import Feedback
from .forms import FeedbackForm

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Create your views here.
def HomeView(request):
    return render(request, 'base.html')


def FaceDetector(request):
    recognizer.read('StaticFiles/recognizer/trainner.yml')
    labels = {'person_name' : 1}
    with open('StaticFiles/pickle/face-labels.pickle', 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()                           # Capture frame-by-frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            id_, conf = recognizer.predict(roi_gray)
            if conf > 45:
                font = cv2.FONT_HERSHEY_SIMPLEX
                name = labels[id_]
                color = (255,255,0)
                stroke = 2
                cv2.putText(frame, name, (x,y), font,1 , color, stroke, cv2.LINE_AA)


            color = (255,0,0)
            stroke = 2
            cv2.rectangle(frame,(x,y), (x+w, y+h), color, stroke)

        cv2.imshow('frame', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    return redirect('/feedback/')

def FaceTrainer(request):
    image_dir = os.path.join(st.BASE_DIR, "StaticFiles")
    
    y_labels = []
    x_train = []
    current_id = 0
    label_ids = {}
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith('png') or file.endswith('jpg') or file.endswith('jpeg'):                
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path))
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                
                id_ = label_ids[label]
                # print(label_ids)
                pil_image = Image.open(path).convert('L')
                size = (550,550)
                final_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(final_image, 'uint8')
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    with open('StaticFiles/pickle/face-labels.pickle', 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save('StaticFiles/recognizer/trainner.yml')

    messages.success(request, 'Model Trainning Completed Successfully')
    return redirect('/')

@login_required
def FeedbackView(request):
    if request.method == "POST":
        form = FeedbackForm(request.POST or None)
        if form.is_valid():
            form.save()
            messages.success(request, 'Your Feedback Means A Lot. Thanks !!!')
            return redirect("/")

    else:
        form = FeedbackForm()

    context = {
        'form' : form
    }
    return render(request, 'feedback.html', context)

@login_required
def FeedbackLogView(request):
    context = {
        'feedback' : Feedback.objects.all()
    }
    return render(request, 'feedback-log.html', context)