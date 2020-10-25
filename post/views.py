from django.shortcuts import render
from .serializers import PostSerializer
from .models import Post
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import subprocess


class PostView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        posts_serializer = PostSerializer(data=request.data)
        if posts_serializer.is_valid():
            print("posts_serializer: ", posts_serializer)
            print("posts_serializer.data: ", posts_serializer.validated_data)
            print("posts_serializer.validated_data['type']: ", posts_serializer.validated_data['type'])
            cmd = 'python3 -u ./post/SSP/test.py --imgs {0} --gpu 0 --cfg ./post/SSP/config/ade20k-hrnetv2.yaml TEST.result test_result/wall/ TEST.checkpoint epoch_0.pth MODEL.object_index 0'.format(posts_serializer.validated_data['room_image'])
            subprocess.call(cmd, shell=True)
            posts_serializer.save()
            print("save successfully!")
            return Response(posts_serializer.data, status=status.HTTP_201_CREATED)
        else:
            print('Error: ', posts_serializer.errors)
            return Response(posts_serializer.errors, status=status.HTTP_400_BAD_REQUEST)