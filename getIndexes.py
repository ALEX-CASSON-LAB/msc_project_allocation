# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:32:55 2022

@author: mchssac7
"""

def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    result = dfObj.isin([value]) # Get bool dataframe with True at positions where the given value exists
    seriesObj = result.any() # Get list of columns that contains the value
    columnNames = list(seriesObj[seriesObj == True].index) # Get list of columns that contains the value
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    return listOfPos     # Return a list of tuples indicating the positions of value in the dataframe