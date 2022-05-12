from http.server import BaseHTTPRequestHandler, HTTPServer
import pickle
import json
import numpy as np

from data.preprocess import preprocess_sentence

class S(BaseHTTPRequestHandler):
    def __init__(self, *args):
        self.clf = pickle.load(open("./models/tfidf_model.pkl", "rb"))
        BaseHTTPRequestHandler.__init__(self, *args)

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self._set_response()

        #sentence = preprocess_sentence("What dies my JS compiler have in common with Pytorch?", vectorizer)
        labels = self.clf.predict(np.array(["What dies my JS compiler have in common with Pytorch?"]).reshape(1, -1))

        response = {
            "Tags" : labels
        }
        json_str=json.dumps(response)
        self.wfile.write(json_str.encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself

        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=S, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print('Stopping httpd...\n')

def serve(port=3333):
    run(port=port)