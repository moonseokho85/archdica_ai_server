from django.contrib import admin
from .models import Post


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ('id', 'email', 'room_image', 'type', 'reference_image', 'wall_mask_image', 'floor_mask_image', 'conversion_image', 'created_at', 'updated_at')

