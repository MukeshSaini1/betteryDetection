import base64

import requests

# The URL of your Flask API endpoint
url = 'http://127.0.0.1:5000/detect'

# Path to the image file you want to test
image_path = 'view_0_Dgl (6).png'  # Replace with your image path

# Send the request to the API
with open(image_path, 'rb') as file:
    response = requests.post(url, files={'file': file})

# Handle the response
if response.status_code == 200:
    results = response.json()

    for result in results:
        image_data = result['image']
        image_filename = result['filename']

        # Decode base64 image data
        with open(f'{image_filename}_annotated.png', 'wb') as f:
            f.write(base64.b64decode(image_data))

        print(f'Annotated image saved as {image_filename}_annotated.png')
else:
    print('Failed to get a response:', response.status_code, response.text)
