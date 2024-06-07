import json
import numpy
import joblib
import time
from azureml.core.model import Model


def init():
    global model
    # Load the model from file into a global object
    model_path = Model.get_model_path("nytaxifareamount_model")
    model = joblib.load(model_path)
    # Print statement for appinsights custom traces:
    print("model initialized" + time.strftime("%H:%M:%S"))


def run(raw_data, request_headers):
    data = json.loads(raw_data)["data"]
    data = numpy.array(data)
    result = model.predict(data)
    # Log the input and output data to appinsights:
    info = {
        "input": raw_data,
        "output": result.tolist()
        }
    print(json.dumps(info))
    print(('{{"RequestId":"{0}", '
           '"TraceParent":"{1}", '
           '"NumberOfPredictions":{2}}}'
           ).format(
               request_headers.get("X-Ms-Request-Id", ""),
               request_headers.get("Traceparent", ""),
               len(result)
    ))

    return {"result": result.tolist()}
