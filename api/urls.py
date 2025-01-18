from django.urls import path

from api.v1.views import SearchView, DataView

urlpatterns = [
    path('search/', SearchView.as_view(), name='search'),
    path('data/', DataView.as_view(), name='data'),
]