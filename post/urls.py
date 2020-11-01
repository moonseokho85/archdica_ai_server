from django.urls import path
from . import views

urlpatterns = [
    path('posts/', views.PostView.as_view(), name='posts_list'),
    path('materials/', views.MaterialListCreateAPIView.as_view(), name='materials_list')
]