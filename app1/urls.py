#from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('',views.home, name ='home'),
    path('demo/',views.demo, name='demo'),
    path('content/',views.content, name='content'),
    path('collaboritive/',views.collaboritive, name='collaboritive'),
    path('showcontent/',views.showcontent, name='showcontent'),
    path('showcollaborative/',views.showcollaborative, name='showcollaborative'),
    path('visual/',views.visual, name='visual'),
    path('visual2/',views.visual2, name='visual2'),
    path('visual3/',views.visual3, name='visual3'),
    path('see/',views.see, name='see'),
    path('about/',views.about, name='about'),
]
