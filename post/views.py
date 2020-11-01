from .models import Post, Materials
from .serializers import PostSerializer, MaterialSerializer, FindSimilarMaterialSerializer
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import subprocess
from .functions.synthesize import synthesize


class ConvertImageAPIView(APIView):
    parser_classes = (FormParser, MultiPartParser)

    def get(self, request, *args, **kwargs):
        posts = Post.objects.all()
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        posts_serializer = PostSerializer(data=request.data)
        if posts_serializer.is_valid():
            posts_serializer.save()

            # Semantic segmentation
            room_image_url = '.' + posts_serializer.data['room_image']
            if posts_serializer.validated_data['type'] == 'wall':
                cmd = 'python3 -u ./post/SSP/test.py --imgs {0} --gpu 0 --cfg ./post/SSP/config/ade20k-hrnetv2.yaml TEST.result ./post/SSP/test_result/wall/ TEST.checkpoint epoch_0.pth MODEL.object_index 0'.format(
                    room_image_url)
                subprocess.call(cmd, shell=True)
            elif posts_serializer.validated_data['type'] == "floor":
                cmd = 'python3 -u ./post/SSP/test.py --imgs {0} --gpu 0 --cfg ./post/SSP/config/ade20k-hrnetv2.yaml TEST.result ./post/SSP/test_result/floor/ TEST.checkpoint epoch_0.pth MODEL.object_index 3'.format(
                    room_image_url)
                subprocess.call(cmd, shell=True)
            else:
                return

            # Synthesize room image and material image
            conversion_image_url = synthesize(room_image_url, posts_serializer.validated_data['type'])

            return Response(posts_serializer.data, status=status.HTTP_201_CREATED)
        else:
            print('Error: ', posts_serializer.errors)
            return Response(posts_serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class MaterialListCreateAPIView(APIView):

    def get(self, request):
        materials = Materials.objects.filter(active=True)
        serializer = MaterialSerializer(materials, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = MaterialSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class FindSimilarMaterialAPIView(APIView):
    parser_classes = (FormParser, MultiPartParser)

    def post(self, request):
        fsm_serializer = FindSimilarMaterialSerializer(data=request.data)
        if fsm_serializer.is_valid():
            fsm_serializer.save()
            return Response(fsm_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(fsm_serializer.data, status=status.HTTP_400_BAD_REQUEST)
