#JSON REQUEST
# import checksum generation utility
import PaytmChecksum

# initialize JSON String
body = "{"\mid\":"\YOUR_MID_HERE\","\orderId\":"\YOUR_ORDER_ID_HERE\"}"

# Generate checksum by parameters we have
# Find your Merchant Key in your Paytm Dashboard at https://dashboard.paytm.com/next/apikeys
paytmChecksum = PaytmChecksum.generateSignature(body, "YOUR_MERCHANT_KEY")
print("generateSignature Returns:" + str(paytmChecksum))



#FOR POST REQUEST
# initialize an Hash/Array
paytmParams = {}

paytmParams["MID"] = "YOUR_MID_HERE"
paytmParams["ORDERID"] = "YOUR_ORDER_ID_HERE"

# Generate checksum by parameters we have
# Find your Merchant Key in your Paytm Dashboard at https://dashboard.paytm.com/next/apikeys
paytmChecksum = PaytmChecksum.generateSignature(paytmParams, "YOUR_MERCHANT_KEY")
print("generateSignature Returns:" + str(paytmChecksum))


#VALIDATE CHECKSUM
# string we need to verify against checksum
body = "{"\mid\":"\YOUR_MID_HERE\","\orderId\":"\YOUR_ORDER_ID_HERE\"}"

#checksum that we need to verify
paytmChecksum = "CHECKSUM_VALUE"

# Verify checksum
# Find your Merchant Key in your Paytm Dashboard at https://dashboard.paytm.com/next/apikeys

isVerifySignature = PaytmChecksum.verifySignature(body, "YOUR_MERCHANT_KEY", paytmChecksum)
if isVerifySignature:
	print("Checksum Matched")
else:
	print("Checksum Mismatched")


#FORM POST REQUEST
paytmParams = dict()
paytmParams = request.form.to_dict()
paytmChecksum = paytmChecksum
paytmChecksum = paytmParams['CHECKSUMHASH']
paytmParams.pop('CHECKSUMHASH', None)

# Verify checksum
# Find your Merchant Key in your Paytm Dashboard at https://dashboard.paytm.com/next/apikeys
isVerifySignature = PaytmChecksum.verifySignature(paytmParams, "YOUR_MERCHANT_KEY",paytmChecksum)
if isVerifySignature:
	print("Checksum Matched")
else:
	print("Checksum Mismatched")
