from django.db import models

# Model for whats in the bowl?
class ProduceImage(models.Model):
    name = models.CharField(max_length=50)
    image = models.ImageField(upload_to='images/')

def __str__(self):
     return self.title
