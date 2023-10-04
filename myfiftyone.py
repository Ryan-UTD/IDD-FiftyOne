import glob
import fiftyone as fo
import fiftyone.zoo as foz
import xml.etree.ElementTree as ET

# Train val split dictionary

splits = ['train', 'val']
split_dictionary = {}

for split in splits:
    
    path = f"C:/Users/ryan/Desktop/thesis/fiftyone/IDD_Detection/{split}.txt"

    with open(path, 'r') as f:
        for line in f:
            split_dictionary[line.replace('\n', '')] = split

# The folder IDD_Detection contains the two folders Annotations and JPEGImages
annotations_path = 'C:/Users/ryan/Desktop/thesis/fiftyone/IDD_Detection/Annotations'

# samples: The list of all samples to create the Dataset with.
# Each sample will contain the image path and its annotations
samples = []

# Read each XML file
for annot_path in glob.glob(annotations_path + '/**/*.xml', recursive=True):
    with open(annot_path, 'r') as f:
        
        # Modify the path to point to the jpg image instead of the XML file
        img_path   = (f
                      .name
                      .replace('/Annotations', '/JPEGImages')
                      .replace('.xml', '.jpg')
                      .replace('\\', '/')
                      )
                
        # Denote the image as part of the Train or Val set
        f_split_id = f.name[len(annotations_path)+1:].replace('.xml', '').replace('\\', '/')
        if f_split_id in split_dictionary.keys():
            split = split_dictionary[f_split_id] # Either 'Train' or 'Val'
        else:
            split = 'Other'
            
        # Create the Sample
        sample = fo.Sample(filepath=img_path, tags=[split])
            
        # List of all detections in the image
        detections = []
        
        # The root of the sample
        r = ET.parse(f).getroot()
        
        """
        r[0]: filename
        r[1]: folder
        r[2]: size [width, height, depth]
        
        Everything after r[2] is a detection in the form of:
        [label, [xmin, ymax, xmax, ymin]]
        """
        
        img_width  = int(r[2][0].text)
        img_height = int(r[2][1].text)
        
        if len(r)>3:
            for i in range(3, len(r)):
                label = r[i][0].text
                xmin = int(r[i][1][0].text)
                ymax = int(r[i][1][1].text)
                xmax = int(r[i][1][2].text)
                ymin = int(r[i][1][3].text)
                
                # The bounding box X, Y, width and height must be scaled
                # within the range [0, 1]
                top_left_x  = xmin / img_width
                top_left_y  = ymin / img_height
                bbox_width  = (xmax-xmin) / img_width
                bbox_height = (ymax-ymin) / img_height
                
                bounding_box = [top_left_x, top_left_y, bbox_width, bbox_height]
        
                detections.append(
                    fo.Detection(label=label, bounding_box=bounding_box)
                )
        
        # Add all detections for the sample
        sample['ground_truth'] = fo.Detections(detections=detections)
        
        # Add the sample to our list of samples
        samples.append(sample)
        


# Create fiftyone dataset
dataset = fo.Dataset("IDD_Detection_5")
dataset.add_samples(samples)

# Launch it
session = fo.launch_app(dataset)
#session.wait() # For use outside of a notebook
