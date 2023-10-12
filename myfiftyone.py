import os
import glob
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz

# xml parsers
from fvcore.common.file_io import PathManager
import xml.etree.ElementTree as ET

def load_idd_instances(dirname: str, split: str):
    """
    Load IDD detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    with PathManager.open(os.path.join(dirname, split + ".txt")) as f:
        fileids = np.loadtxt(f, dtype=str)

    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
        jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

        tree = ET.parse(anno_file)

        r = {
            "file_name": jpeg_file,
            "image_id": fileid,
            "height": int(tree.findall("./size/height")[0].text),
            "width": int(tree.findall("./size/width")[0].text),
        }

        # Create the Sample
        sample = fo.Sample(filepath=r["file_name"], tags=[split])

        instances = []

        for obj in tree.findall("object"):
            cls = obj.find("name").text
            bbox = obj.find("bndbox")
            bbox = [float(bbox.find(x).text) for x in ["xmin", "ymin", "xmax", "ymax"]]
            top_left_x  = bbox[0] / r["width"]
            top_left_y  = bbox[1] / r["height"]
            bbox_width  = (bbox[2] - bbox[0]) / r["width"]
            bbox_height = (bbox[3] - bbox[1]) / r["height"]
            
            bbox = [top_left_x, top_left_y, bbox_width, bbox_height]            
            instances.append(
                fo.Detection(label=cls, bounding_box=bbox)
            )
        sample['ground_truth'] = fo.Detections(detections=instances)
        
        dicts.append(sample)
    return dicts

if __name__ == '__main__':
    splits = ['train', 'val']
    base_dir = "/archive/datasets/IDD_Detection/"

    samples = []

    for split in splits:
        split_samples = load_idd_instances(dirname=base_dir, split=split)
        samples.extend(split_samples)
    
    # Create fiftyone dataset
    dataset = fo.Dataset("IDD_Detection")
    dataset.add_samples(samples)

    # Launch it
    session = fo.launch_app(dataset, remote=True, port=9900)
    session.wait() # For use outside of a notebook
