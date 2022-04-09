# A template for a simple server to recv images from esp 
from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import os
import socket

DATASET_NAME = "fruit_test"
DATASETS_PATH = "assets/" + DATASET_NAME + "/images/" # When opening from the main project directory

FILL_SIZE = 3 # Zero Padding for filenames 
FILE_EXTENSION_LEN = len(".jpg")
FILE_IDENTIFIER_LEN = FILL_SIZE + FILE_EXTENSION_LEN


class S(BaseHTTPRequestHandler):
    
    total_images = 0
    image_classes = {}

    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):

        content_length = int(self.headers['Content-Length']) # <--- Gets the size of data
        post_data = self.rfile.read(content_length) # <--- Gets the data itself
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("POST request for {}".format(self.path).encode('utf-8'))

        class_name = input("Enter the object class: ")
        S.total_images += 1
        S.update_Class(class_name)

        class_name += str(S.image_classes.get(class_name)).zfill(FILL_SIZE) + ".jpg"

        f = open(DATASETS_PATH + class_name, "xb")
        f.write(post_data)

        print("Total images: " + str(S.total_images))
        print(S.image_classes)

    def count_Images(self):
        
        #Add a function to number in each class
        items = os.listdir(DATASETS_PATH)

        for item in items:
            if os.path.isfile(os.path.join(DATASETS_PATH, item)):
                S.total_images += 1  

        print("Total images in the dataset: " +  str(S.total_images))
    
    def count_Classes(self):

        items = os.listdir(DATASETS_PATH)

        # Loop through every file in DATASETS_PATH
        for item in items:
            if os.path.isfile(os.path.join(DATASETS_PATH, item)):
                # This is a funky way to chop of the file extension and number 
                class_name = item[:-(FILE_IDENTIFIER_LEN)]
                S.update_Class(class_name)

        print(S.image_classes)

    def update_Class(class_name):

        class_value = S.image_classes.get(class_name)

        if class_value == None:
            print("Adding Class: " + class_name) 
            S.image_classes.update({class_name: 1})  
        else:
            print("Updating Class: " + class_name) 
            class_value += 1
            S.image_classes.update({class_name: class_value})
            

def init_dataset():
    if os.path.exists(DATASETS_PATH):
        print("Adding to dataset at: " + DATASETS_PATH)
    else:
        os.makedirs(DATASETS_PATH)
        print("Creating dataset: " + DATASETS_PATH)


def run(server_class=HTTPServer, handler_class=S, port=80):

    init_dataset()

    # These two functions kinda have redundant computation, could fix? 
    handler_class.count_Images(handler_class) 
    handler_class.count_Classes(handler_class)

    
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info('Starting httpd...\n')

    '''
    Cant trust this I guess :(
    # Just a little info to make my life easier 
    host = socket.gethostname()
    ip = socket.gethostbyname(host)
    print('Host: ', host)
    print('IP: ', ip)
    '''
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

    logging.info('Stopping httpd...\n')
    

run() 