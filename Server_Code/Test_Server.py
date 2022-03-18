# A template for a simple server to recv images from esp 
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import os
import socket

class S(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):

        # Temporary fix to remove file 
        # Will add in a more suitable naming scheme later when I collect data 
        if os.path.exists("tst.jpg"):
            os.remove("tst.jpg")
        else:
         print("The file does not exist") 

        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))

        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))
        f = open("tst.jpg", "xb")
        f.write(post_data)

def run(server_class=HTTPServer, handler_class=S, port=80):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')

    # Just a little info to make my life easier 
    host = socket.gethostname()
    ip = socket.gethostbyname(host)
    print('Host: ', host)
    print('IP: ', ip)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info('Stopping httpd...\n')

run() 