from rest_framework import serializers
from .models import Conversion, Materials, FindSimilarMaterial


class ConversionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversion
        fields = '__all__'
        example = {
            'email': 'archdica@example.com',
            'room_image': 'image.jpg',
            'type': 'wall',
            'reference_image': 'reference.jpg',
        }


class MaterialSerializer(serializers.ModelSerializer):
    class Meta:
        model = Materials
        fields = '__all__'


class FindSimilarMaterialSerializer(serializers.ModelSerializer):
    class Meta:
        model = FindSimilarMaterial
        fields = '__all__'

