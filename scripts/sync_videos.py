


import openpyxl

# wb = openpyxl.load_workbook("video_metadata.xlsx") # creates a workbook object
# ws = wb["videos"] # grab the sheet videos
# print (ws["A1"].value) # test - name column

# headers = [cell.value for cell in ws[1]] # ws[1] is the first row so looping of each value (unpacking each object) in the row gives the headers of each column
# print(headers)

# rows = ws.iter_rows(min_row=2, values_only=True) # a sequence of tuples (each sequience is a row), where each tuple holds a mix of different data types (values for eahc row)
# for row in rows:
#     print(row)

def load_video_row(xlsx_path): # returns a list of dicts, one per video row
    wb = openpyxl.load_workbook(xlsx_path) # creates a workbook object
    ws = wb["videos"] # grab the sheet videos
    headers = [cell.value for cell in ws[1]] # values
    rows = ws.iter_rows(min_row=2, values_only=True) # row min skip row 1 with headers # generator
    return [dict(zip(headers, row)) for row in rows] # returns a dictionary same as the loop version: zip each row against headers, wrap in dict, collect


# call function
print(load_video_row("video_metadata.xlsx"))

