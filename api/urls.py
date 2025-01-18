from django.urls import path, include
from rest_framework.routers import DefaultRouter

from api.v1.views import SearchView, DataView

router = DefaultRouter()
router.register(r'search', SearchView, basename='search')
router.register(r'data', DataView, basename='data')

urlpatterns = [
    path('v1/', include(router.urls)),
]