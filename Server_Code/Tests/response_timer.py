import os
import time
import numpy as np
import matplotlib.pyplot as plt
import requests

DATASET_PATH = "../../assets/datasets/fruit_train02/images/"
SERVER_URL = 'http://10.0.0.118:8000/produce_detector/'
S_TO_NS = 1_000_000_000 # Conversion factor
TEST_RUNS = 100


def time_elapsed(t_finish: int, t_start: int) -> int:
        """ Calculate elapsed time and return as an int"""
        t_elapsed = (t_finish - t_start)
        return t_elapsed

def time_rsp(im: str) -> int:
    """ Time a single server response """
    cur_image = open(DATASET_PATH + im, "rb")
    t_start = time.time_ns()
    requests.post(SERVER_URL , data=cur_image)
    t_finish = time.time_ns()
    t_rsp = time_elapsed(t_finish, t_start)
    return t_rsp / S_TO_NS

def test_rsp_time() -> None:
    """ Send server a series of images and record the time it takes for each response """
    
    image_files = os.listdir(DATASET_PATH)

    num_images = len(image_files)

    print(f"Starting response time test with {num_images} images and looping {TEST_RUNS} times!")

    times = []
    for i in range(TEST_RUNS):
        t_start = time.time_ns()
        times += [time_rsp(im) for im in image_files]
        t_finish = time.time_ns()

    times = np.asarray(times)

    total_time = time_elapsed(t_finish, t_start) / S_TO_NS
    mean_times = np.mean(times)
    std_times = np.std(times)

    print(f"Total time: {total_time:.6f}s")
    print(f"Mean response time: {mean_times:.6f}s")
    print(f"Standard deviation of response times: {std_times:.6f}")
    

    hx, hy, _ = plt.hist(times, bins=50, color="red")

    plt.ylim(0.0,max(hx)+0.05)
    plt.title("Histogram of Server response times")
    plt.grid()
    plt.show()
    
    print("Finished Testing")

    return


test_rsp_time()