from nturl2path import url2pathname
from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('add', views.add, name='add'),
    path('aqi', views.aqi, name='aqi'),
    path('content', views.content, name='content'),
    path('live_aqi', views.live_aqi, name='live_aqi'),
    path('new', views.new, name='new')
    

]