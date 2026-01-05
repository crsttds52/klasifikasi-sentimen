from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    
    path('about/', views.about_view, name='about'),
    
    path('scrape/', views.scrape_view, name='scrape'),
    
    path('preprocess/', views.preprocess_view, name='preprocess'),
    
    path('results/', views.results_view, name='results'),
]