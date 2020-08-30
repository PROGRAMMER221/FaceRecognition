from django import forms
from .models import Feedback
class FeedbackForm(forms.ModelForm):
    class Meta:
        model = Feedback
        fields = ('name', 'detector', 'comment')
        widgets = {
            'name' : forms.TextInput(attrs={
                'class' : 'form-control'
            }),

            'detector' : forms.CheckboxInput(attrs={
                'class' : 'form-control'
            }),

            'comment' : forms.Textarea(attrs={
                'class' : 'form-control'
            })
        }