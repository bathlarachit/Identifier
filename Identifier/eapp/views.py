from django.shortcuts import render
from django.urls import reverse_lazy,reverse
from django.views.generic import TemplateView,CreateView
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input,decode_predictions
import numpy as np
from eapp import forms
# Create your views here.
from eapp import models
def Home(request):

        return render(request,'eapp/base.html')

def imageprocess(request):
    form=forms.ImageUploadForm(request.POST,request.FILES)
    if form.is_valid():
        handle_uploaded_file(request.FILES['image'])
        model=ResNet50(weights='imagenet')
        img_path ='img.jpg'
        img=image.load_img(img_path,target_size=(224,224))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        preds=model.predict(x)
    #    print('Predicted:',decode_predictions(preds,top=3)[0])

        html=decode_predictions(preds,top=3)[0]
        res=[]
        for e in html:
            res.append((e[1],np.round(e[2]*100,2)))

        return render(request,'eapp/result.html',{'res':res})
    return render(request,'eapp/result.html')

def handle_uploaded_file(f):
    with open('img.jpg','wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
