import mysql.connector
import yaml

def get_connection(): # defining a function to use in any script to connect with database sql
    with open('config.yaml') as f: # converting config.yaml into a dict
        cfg = yaml.safe_load(f)
    return mysql.connector.connect(**cfg['database']) # grabs the database key from dict / ** unpack meaning dict becomes var (eg. {host:'localhost} -> host='localhost')
