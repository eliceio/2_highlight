from django.conf.urls import url
from . import views
from highlight import settings
from django.conf.urls.static import static

urlpatterns = [
	url(r'^index/$', views.index, name='index'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
