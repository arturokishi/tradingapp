from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),
    # ✅ CORRECT - include the main core.urls which already has all your trading paths
    path('trading/', include('core.urls')),
]

