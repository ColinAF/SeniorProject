from django.shortcuts import render
from django.http import HttpResponse

def index(request):
    
    post = False 

    if request.method == 'POST':
        print("Sombody posted!")
        post = True
    
    else:
        print("Waiting for a POST!")

    context = {'posted' : post}
    return render(request, 'produce_detector/index.html', context)
