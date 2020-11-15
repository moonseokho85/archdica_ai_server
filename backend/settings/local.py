from .base import *

DEBUG = True

ALLOWED_HOSTS = []

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'static')
# STATICFILES_STORAGE = 'pipeline.storage.PipelineCachedStorage'  # Local, 즉 DEBUG=True 일 경우 pipeline 사용

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')