
# IMPORTS
import torch
from torch.utils.data import Dataset #import Dataset from PyTorch's data utilities.import os
import mysql.connector
import cv2
 
class AquaMindDataset(Dataset): # naming my own class (import pytorch class to inherit)    
    def __init__(self):
        conn = mysql.connector.connect( # connect with sql database
            host='localhost',
            port=3306,
            user='root',
            password='aquamind',
            database='aquamind'
        )   
        
        cursor = conn.cursor() 
        cursor.execute("SELECT id, frame_path FROM frames") # preparation of request

        self.frames = cursor.fetchall() # take in request








