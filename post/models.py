from django.db import models
import uuid

TYPE_CHOICES = (
    ('wall', 'wall'),
    ('floor', 'floor')
)


class Post(models.Model):
    email = models.EmailField(max_length=254, blank=False)
    room_image = models.ImageField(upload_to="images/room/%Y/%m/%d", blank=False)
    type = models.CharField(max_length=20, blank=False, choices=TYPE_CHOICES)
    reference_image = models.ImageField(upload_to="images/reference/%Y/%m/%d", blank=False)
    wall_mask_image = models.ImageField(upload_to="images/wall_mask/%Y/%m/%d", blank=True)
    floor_mask_image = models.ImageField(upload_to="images/floor_mask/%Y/%m/%d", blank=True)
    conversion_image = models.ImageField(upload_to="images/conversion/%Y/%m/%d", blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.email