from django.shortcuts import render
from . import test , testTeach
from mainSite.testTeach import teach_gesture
from mainSite.test import recognize_gesture
from django.http import StreamingHttpResponse 



def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        

# Create your views here.
def index(request):

    return render(request, 'index.html')


def teachIndex(request):
     return render(request, 'teach.html')

def teach(request):

    gesture = teach_gesture()  # Create an instance of teach_gesture class

    return StreamingHttpResponse(gen(gesture), content_type='multipart/x-mixed-replace; boundary=frame')

        


def recognizeIndex(request):

    return render(request, 'recognize.html')
    #return StreamingHttpResponse(gen(recognize_gesture()),content_type='multipart/x-mixed-replace; boundary=frame')
def recognize(request):
    
    cam = recognize_gesture()
    return StreamingHttpResponse(gen(cam), content_type='multipart/x-mixed-replace; boundary=frame')
    



