from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render

def index(request):
	print("Hi AWS!!!!!!!!!!!!!!!!")
	#return HttpResponse("Hello, World Local Test")
	return HttpResponseRedirect("request/index")