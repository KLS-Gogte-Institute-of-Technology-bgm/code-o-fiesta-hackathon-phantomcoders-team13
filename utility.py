import requests
import json

def function(amount):
	"""
	Function returns an unrendered QR code in SVG format
	"""
	headers = {
	    'Content-type': 'application/json',
	}

	data = {'name':'Prathamesh Koranne' , 'vpa':'9742163501' ,'amount': amount }

	response = requests.post('https://upiqr.in/api/qr', headers=headers, data=data)

	return response.text