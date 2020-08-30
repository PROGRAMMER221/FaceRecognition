from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$',views.HomeView, name="home"),
    url(r'^face-detector/$', views.FaceDetector, name="face-detector"),
    url(r'^face-trainer/$', views.FaceTrainer, name="face-trainer"),
    url(r'^feedback/$', views.FeedbackView, name='feedback'),
    url(r'^feedback-log/$', views.FeedbackLogView, name='feedback-log'),
]