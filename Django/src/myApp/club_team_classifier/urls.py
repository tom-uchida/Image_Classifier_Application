from django.urls import path

from . import views

app_name = 'club_team_classifier'
urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict, name='predict'),
]