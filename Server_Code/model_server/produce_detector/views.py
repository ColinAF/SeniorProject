from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse

from .forms import *
from .models import ProduceImage
from django.template import loader
from .detector import ObjectDetector

import time

odm = ObjectDetector()

def index(request):

    if request.method == 'POST':
        # Make sure the image updates properly
        ProduceImage.objects.all().delete()
        im = request.body
        image_name = 'produce_bowl' + time.strftime("_%H_%M_%S", time.gmtime()) + '.jpg'
        save_image(im, image_name)
        produce = odm.run_model(image_name)
        print(produce)

        for i in produce:
            pst = ProduceImage(object_class=i, qty=produce[i])
            pst.save()
        # Should probably clean up the images eventually! 
    else:
       # Default Image
       # How can I make this display only 
       image_name = 'produce_bowl'
 
    bowl = ProduceImage.objects.all().values()
    template = loader.get_template('produce_detector/index.html')
    
    print(bowl)
    context = {'bowl' : bowl, 'image_name' : image_name}

    return HttpResponse(template.render(context, request)) # Make it so that things update everywhere on a post!

def save_image(f, name):
    # Save somewhere new each time! 
    path = 'media/images/' + name
    with open(path, 'wb+') as destination:
        destination.write(f)


