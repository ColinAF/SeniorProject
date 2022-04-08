### CVS Data Logger ###
import csv 

class CSVLogger():
    # Header is an optional dict argument that can populate and empty CSV file with a header
    def __init__(self, filename, header=None):
        # Test is the file exists (For now we assume it won't)
        training_stats = open(filename, 'w')
        self.stats_writer = csv.writer(training_stats)

        if header is not None:
            self.stats_writer.writerow(header)

    # Log expects a dict representing a new row
    def log(self, data):
        self.stats_writer.writerow(data)
        

### CVS Data Logger ###