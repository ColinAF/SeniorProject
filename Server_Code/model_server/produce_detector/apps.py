from django.apps import AppConfig
from .detector import ObjectDetector


class ProduceDetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'produce_detector'

        