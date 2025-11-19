from evaluation.coco_json_evaler import CocoJsonEvaler
from evaluation.flickr8k_json_evaler import Flickr8kJsonEvaler

__factory = {
    "COCO": CocoJsonEvaler,
    'FLICKR8K': Flickr8kJsonEvaler,
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown Evaler:", name)
    return __factory[name](*args, **kwargs)
