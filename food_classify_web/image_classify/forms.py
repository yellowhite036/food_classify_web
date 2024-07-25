from django import forms
from .models import UploadedImage

class ImageUploadForm(forms.ModelForm):
    price = forms.DecimalField(label='Enter Price', required=False)

    class Meta:
        model = UploadedImage
        fields = ['image', 'price']