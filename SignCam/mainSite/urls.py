from django.contrib import admin
from django.urls import path
from . import views
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index),
    path('Index', views.index, name='index'),
    path('teachIndex', views.teachIndex, name='teachIndex'),
    path('teach/', views.teach, name='teach'),
    
    path('recognizeIndex', views.recognizeIndex, name='recognizeIndex'),
    path('recognize', views.recognize, name='recognize')
      # Add this line for the index view
    
]