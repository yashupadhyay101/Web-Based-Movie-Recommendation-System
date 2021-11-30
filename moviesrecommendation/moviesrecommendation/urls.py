"""moviesrecommendation URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from . import views
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('',include('app1.urls')),
    path('demo/',include('app1.urls')),
    path('content/',include('app1.urls')),
    path('content_1/',include('app1.urls')),
    path('collaboritive/',include('app1.urls')),
    path('showcontent/',include('app1.urls')),
    path('showcontent_1/',include('app1.urls')),
    path('showcollaborative/',include('app1.urls')),
    path('about/',include('app1.urls')),
    path('visual/',include('app1.urls')),
    path('visual2/',include('app1.urls')),
    path('visual3/',include('app1.urls')),
    path('see/',include('app1.urls')),
    path('admin/', admin.site.urls),
]

urlpatterns += staticfiles_urlpatterns()