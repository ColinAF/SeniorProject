from django.db import models

# Model for whats in the bowl
class ProduceImage(models.Model):
    object_class = models.CharField(max_length=50)
    qty = models.CharField(max_length=50) # Quantity
    

def __str__(self):
     return self.title
