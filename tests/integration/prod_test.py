import requests
import json

test_sample = json.dumps({'data': [[-73.973320,40.76380,-73.981430,40.743835,1],
                                   [-73.994662,40.753627,-73.989927,40.757178,2],
                                   [-73.920513,40.743622,-73.985523,40.752782,3]
                                   [-73.999722,40.743675,-73.988065,40.728742,4]]})
test_sample = str(test_sample)

def test_ml_service(scoreurl, scorekey):
    assert scoreurl != None

    if scorekey is None:
        headers = {'Content-Type':'application/json'}
    else:
        headers = {'Content-Type':'application/json', 'Authorization':('Bearer ' + scorekey)}

    resp = requests.post(scoreurl, test_sample, headers=headers)
    assert resp.status_code == requests.codes.ok
    assert resp.text != None
    assert resp.headers.get('content-type') == 'application/json'
    assert int(resp.headers.get('Content-Length')) > 0