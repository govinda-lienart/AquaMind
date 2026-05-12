
# IMPORTS
import torch
from torch.utils.data import Dataset #import Dataset from PyTorch's data utilities.import os
import mysql.connector
import cv2
 
class AquaMindDataset(Dataset): # naming my own class (import pytorch class to inherit)    
    def __init__(self):
        self.conn = mysql.connector.connect( # connect with sql database - self makes it a property of the object, available to every method for the lifetime of the object. Without self, it's just a local variable that disappears when the method ends.            host='localhost',
            port=3306,
            user='root',
            password='aquamind',
            database='aquamind'
        )   
        
        cursor = self.conn.cursor()  # each time I run a query I need a cursor. 
        cursor.execute("SELECT id, frame_path FROM frames") # preparation of request
        self.frames = cursor.fetchall() # take in request and stores a list of tuples, one per frame  (66, 'frames/frames_IMG_9856_20260506_0957/frame_240.png'),  (67, 'frames/frames_IMG_9856_20260506_0957/frame_300.png'),  (68, 'frames/frames_IMG_9856_20260506_0957/frame_360.png')
    
    def __len__(self): 
         return len(self.frames)   

    def __getitem__(self, idx):     
        frame_id, frame_path = self.frames[idx]  #self.frames[idx] pulls the tuple at position idx from the list,  and frame_id, frame_path = ... unpacks its two elements into     separate variables.                                             
        image = cv2.imread(frame_path) #imread = image read =  the model needs the pixel values as numbers to do matrix operations during training. The NumPy array is just a grid of  numbers (height × width × colour channels). PyTorch will then  convert it to a tensor to run through the CNN.  
        image = torch.from_numpy(image) # converting into pytorch tensor for neural network -   multi-dimensional grid of numbers
        cursor = self.conn.cursor()                                                     
        cursor.execute("SELECT * FROM annotations WHERE frame_id = %s", (frame_id,))
        annotations = cursor.fetchall()
        image = torch.from_numpy(image) # converting into pytorch tensor for neural network -   multi-dimensional grid of numbers
        return image, annotations


dataset = AquaMindDataset()

# some testing of the class but will not be used by pytorch libary (the library will call it automatically) 
print(dataset) # that just gives the position in memory <__main__.AquaMindDataset object at 0x105ee1010>
print (dataset[0]) # returns from frames tables the first row(62) returns frames array and info for that row pulled from annotaionts...array([[[171, 182, 187]....[(54, 62, 0, 'fish', 0.392133, 0.63347, 0.0912474, 0.0677618, datetime.datetime(2026, 5, 8, 15, 27, 25))]),

dataset.__getitem__(0)  # same results as above
print(len(dataset))   #output is: 61                                


