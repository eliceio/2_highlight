from django.db import models

# Create your models here.

def upload_to_input(instance, filename):
    return 'file/input/%s' % (str(instance.pk)+"_"+filename)

def upload_to_output(instance, filename):
    return 'file/output/%s' % ("out_"+str(instance.pk)+"_"+filename)

def upload_to_output2(instance, filename):
    return 'file/output2/%s' % ("out_"+str(instance.pk)+"_"+filename)

def upload_to_img(instance, filename):
    return 'file/img/%s' % (str(instance.pk)+"_"+filename)

class Face_img(models.Model):
    img = models.FileField(blank=True, null=True, upload_to=upload_to_img)
    fname = models.CharField(max_length=30, null=True)

class Highlight(models.Model):
    f_img = models.ManyToManyField(Face_img, blank=True, null=True)
    file_in = models.FileField(blank=True, null=True, upload_to=upload_to_input)
    fname = models.CharField(max_length=30, null=True)
    file_out1 = models.FileField(blank=True, null=True, upload_to=upload_to_output)
    file_out2 = models.FileField(blank=True, null=True, upload_to=upload_to_output2)
    status = models.IntegerField(default=0)