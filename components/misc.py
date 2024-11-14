import yaml
from datetime import datetime

class Log():
    def __init__(self):
        self.log_file = f"./logs/log{datetime.now().strftime('%m-%d--%H-%M')}.txt"
        
    def log_default(self, epoch, totReward, epsilon, score, mode="+a"):
        with open(self.log_file, mode) as log:
            log.write(f"{datetime.now()}: epoch: {epoch} | totalReward = {totReward} | epsilon = {epsilon} | pipes passed = {score}\n")

    def log_parameters(self, mode="a"):
        with open("hyperparameters.yml", "r") as parameters_file:
            parameters = yaml.safe_load(parameters_file)

        with open(self.log_file, mode) as f:
            for key, value in parameters.items():
                f.write(f"{key}: {value}\n")
        
            f.write("\n")