from .models import Post, Materials
from .serializers import PostSerializer, MaterialSerializer, FindSimilarMaterialSerializer
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
import subprocess
from .functions.synthesize import synthesize
from .functions.ssp_execute import execute


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

            try:
                execute(room_image_url)
            except Exception as e:
                print('Error: ', e)

            wall_mask_url = "https://{0}.s3.{1}.amazonaws.com/{2}".format(AWS_BUCKET_NAME, AWS_DEFAULT_REGION, img_name)
            floor_mask_url = "https://{0}.s3.{1}.amazonaws.com/{2}".format(AWS_BUCKET_NAME, AWS_DEFAULT_REGION, img_name)

            # Synthesize room image and material image
            conversion_image_url = synthesize(room_image_url, posts_serializer.validated_data['type'])

            newdict = {
                'conversion_image_url': conversion_image_url,
                'wall_mask_url': wall_mask_url,
                'floor_mask_url': floor_mask_url
            }
            newdict.update(posts_serializer.data)

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
