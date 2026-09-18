


import openpyxl

wb = openpyxl.load_workbook("video_metadata.xlsx") # creates a workbook object
ws = wb["videos"] # grab the sheet videos
print (ws["A1"].value) # test - name column

