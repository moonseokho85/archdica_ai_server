from rest_framework import serializers
from .models import Post, Materials, FindSimilarMaterial


class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = '__all__'


class MaterialSerializer(serializers.ModelSerializer):
    class Meta:
        model = Materials
        fields = '__all__'


class FindSimilarMaterialSerializer(serializers.ModelSerializer):
    class Meta:
        model = FindSimilarMaterial
        fields = '__all__'

