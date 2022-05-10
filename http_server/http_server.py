from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
from os import curdir, sep

from urllib.parse import parse_qs

hostName = "localhost"
serverPort = 8080

class Server(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def _set_headers(self, data):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(f"Data is: {data}, Specifically input is: {data['input']}".encode('utf-8'))


    def do_GET(self):

        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))

        # If root path is called, then we serve the index.html file back.
        self._set_response()
        if(self.path == "/"):
            f = open(curdir + sep + "./index.html", 'rb')
            self.wfile.write(f.read())
        else:
            self.wfile.write("GET request for {}\n".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                     str(self.path), str(self.headers), post_data.decode('utf-8'))

        if(self.path == "/predict"):
            # TODO: Call model here with provided data (should be in post_data I think)
            # For now just echoes back the provided input data
            parsed_data = parse_qs(post_data.decode('utf-8'))
            logging.info(f"(parsed) received data: {parsed_data}")

            # send the message back
            self._set_headers(parsed_data)
            pass
        else:
            self._set_response()
            self.wfile.write("POST request for {}\n".format(self.path).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=Server, port=8080):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')


if __name__ == '__main__':
    from sys import argv

    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()