from google.cloud import storage
import pickle
import sklearn

BUCKET_NAME = "cloud_fct_michaelf"
MODEL_FILE = "model.pkl"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
my_model = pickle.loads(blob.download_as_string())

def handle_request(request):
    """handle """
    request_json = request.get_json()
    return knn_classifier(request_json)
    
    
def knn_classifier(request_json):
    """ will to stuff to your request """
    if request_json and 'input_data' in request_json:
        data = request_json['input_data']
        input_data = list(map(int, data.split(',')))
        prediction = my_model.predict([input_data])
        return f'Belongs to class: {prediction}'
    else:
        return 'No input data received'

def test_function():
    print(knn_classifier({"input_data": "1,1,1,1"}))

if __name__ == "__main__":
    test_function()