from django.shortcuts import render
from django.http import HttpResponse
from mainSite import test
def index(request):
    return render(request, 'index.html')
def recognize(request):
    # Code to capture video feed and process frames for gesture recognition
    # Call your hand gesture recognition system and get the recognized gesture
    
    return render(request, 'recognize_gesture.html')