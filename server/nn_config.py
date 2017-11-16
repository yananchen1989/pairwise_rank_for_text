import json

class JSONObject:
     def __init__(self, d):
         self.__dict__ = d


class NNConfig(object):
    def __init__(self, path):
        with open(path, 'r') as myfile:
            data=myfile.read().replace('\n', '')
        self.config = json.loads(data, object_hook=JSONObject)

    def get_config(self):
        return self.config

if __name__ == "__main__":
    modelConfig = NNConfig("nn_config.json")
    configOb = modelConfig.get_config()
    print(configOb.data_path)
