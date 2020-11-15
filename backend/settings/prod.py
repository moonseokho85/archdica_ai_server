from .base import *

DEBUG = False

ALLOWED_HOSTS = ['*']

# AWS Configuration

AWS_REGION = "ap-northeast-2"  # AWS 지역

DEFAULT_FILE_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'
STATICFILES_STORAGE = 'storages.backends.s3boto3.S3Boto3Storage'

AWS_ACCESS_KEY_ID = config('AWS_ACCESS_KEY_ID')  # 액세스 키
AWS_SECRET_ACCESS_KEY = config('AWS_SECRET_ACCESS_KEY')  # 비밀 액세스 키

AWS_STORAGE_BUCKET_NAME = "archdica-ai-bucket"  # 버킷 이름
AWS_S3_CUSTOM_DOMAIN = '%s.s3.%s.amazonaws.com' % (AWS_STORAGE_BUCKET_NAME, AWS_REGION)

# Static Setting
AWS_STATIC_LOCATION = 'static'
STATIC_URL = "https://%s/%s" % (AWS_S3_CUSTOM_DOMAIN, AWS_STATIC_LOCATION)
STATICFILES_STORAGE = 'storages.backends.s3boto.S3BotoStorage'

# Media Setting
AWS_PUBLIC_MEDIA_LOCATION = 'media/public'
DEFAULT_FILE_STORAGE = 'backend.storages.PublicMediaStorage'

AWS_PRIVATE_MEDIA_LOCATION = 'media/private'
PRIVATE_FILE_STORAGE = 'backend.storages.PrivateMediaStorage'
