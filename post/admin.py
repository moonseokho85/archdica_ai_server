from django.contrib import admin
from .models import Conversion, Materials, FindSimilarMaterial


@admin.register(Conversion)
class ConversionAdmin(admin.ModelAdmin):
    list_display = ('id', 'email', 'room_image', 'type', 'reference_image', 'wall_mask_image', 'floor_mask_image', 'conversion_image', 'created_at', 'updated_at')


@admin.register(Materials)
class MaterialAdmin(admin.ModelAdmin):
    list_display = ('brand', 'sub_brand', 'index', 'type', 'url', 'created_at', 'updated_at')


@admin.register(FindSimilarMaterial)
class FindSimilarMaterialAdmin(admin.ModelAdmin):
    list_display = ('email', 'room_image', 'type', 'created_at', 'updated_at')



