# system libs
import os

# django libs
from .models import Conversion, Materials
from .serializers import ConversionSerializer, MaterialSerializer, FindSimilarMaterialSerializer
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from .functions.synthesize import synthesize
from .functions.ssp_execute import execute
from decouple import config

# config
AWS_DEFAULT_REGION = config('AWS_DEFAULT_REGION')


class ConvertImageAPIView(APIView):

    """
        이미지를 변환해 주는 API
        ---
        # 내용
            - email : 이메일
            - room_image : 방 이미지
            - type : 유형(벽 or 바닥)
            - reference_image : 참조 이미지(벽지 or 바닥재)
    """

    parser_classes = (FormParser, MultiPartParser)

    def get(self, request, *args, **kwargs):
        posts = Conversion.objects.all()
        serializer = ConversionSerializer(posts, many=True)
        return Response(serializer.data)

    def post(self, request, *args, **kwargs):
        posts_serializer = ConversionSerializer(data=request.data)
        if posts_serializer.is_valid():
            posts_serializer.save()
            print('posts_serializer.data: ', posts_serializer.data)

            # Semantic segmentation
            room_image_url = '.' + posts_serializer.data['room_image']
            reference_image_url = '.' + posts_serializer.data['reference_image']

            try:
                execute(room_image_url)
            except Exception as e:
                print('Error: ', e)

            room_image_name = os.path.basename(room_image_url)

            wall_mask_url = "https://{0}.s3.{1}.amazonaws.com/{2}".format('wall-mask', AWS_DEFAULT_REGION,
                                                                          room_image_name)
            floor_mask_url = "https://{0}.s3.{1}.amazonaws.com/{2}".format('floor-mask', AWS_DEFAULT_REGION,
                                                                           room_image_name)

            # Synthesize room image and material image
            conversion_image_url = synthesize(room_image_url, reference_image_url,
                                              posts_serializer.validated_data['type'])

            newdict = {
                'conversion_image_url': conversion_image_url,
                'wall_mask_url': wall_mask_url,
                'floor_mask_url': floor_mask_url
            }
            newdict.update(posts_serializer.data)  # update previous data

            return Response(newdict, status=status.HTTP_201_CREATED)
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
