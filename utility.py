import requests
import json

def qrgen(amount):
	"""
	Function returns an unrendered QR code in SVG format
	"""
	headers = {
	    'Content-type': 'application/json',
	}

	data = {'name':'Prathamesh Koranne' , 'vpa':'9742163501@paytm' ,'amount': amount }

	response = requests.post('https://upiqr.in/api/qr?format=png', headers=headers, data=json.dumps(data))
	with open('./static/sample.png', 'wb') as f:
		f.write(response.content)

	
qrgen(900)		

# import requests

# headers = {
#     'Content-type': 'application/json',
# }

# data = '{ "name" : "Santanu Sinha", "vpa": "santanu@ybl"}'

# response = requests.post('https://upiqr.in/api/qr', headers=headers, data=data)