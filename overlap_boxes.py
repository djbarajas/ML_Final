import numpy as np

def isInside(bigRect, smallRect):
    '''
    Checks if the small rectangle is completely inside the big rectangle
    
    @param bigRect is an array with: [x, y, width, height]
    @param smallRect is an array with: [x, y, width, height]
    @return true if the small rectangle is inside; false otherwise
    '''
    big_x = bigRect[0]
    big_y = bigRect[1]
    big_width = bigRect[2]
    big_length = bigRect[3]
    small_TL_x = smallRect[0]
    small_TL_y = smallRect[1]
    small_width = smallRect[2]
    small_length = smallRect[3]
    
#     print("big bottom left:", big_x, big_y - big_width)
#     print("big top right: ", big_x + big_length, big_y)
    
    small_BL_x = small_TL_x
    small_BL_y = small_TL_y - small_width
    
#     print("small bottom left: ", small_BL_x, small_BL_y)
    
    small_TR_x = small_TL_x + small_length
    small_TR_y = small_TL_y
    
#     print("small Top Right corner: ", small_TR_x, small_TR_y)
    
    # small rectangle's top left corner
    if (pointIsInside(big_x, big_y, big_length, big_width, small_TL_x, small_TL_y)):
        if (pointIsInside(big_x, big_y, big_length, big_width, small_BL_x, small_BL_y)):
            if(pointIsInside(big_x, big_y, big_length, big_width, small_TR_x, small_TR_y)):
                return True
    
    return False

def pointIsInside(big_x, big_y, big_length, big_width, small_x, small_y):
    '''
    Determines if a small rectangle's random corner is inside the bigger rectangle
    
    @param big_x the x-coordinate of the top left corner of big rectangel
    @param big_y the y-coordinate of the top left corner of the big rectangle
    @param big_length the length of the big rectangle
    @param big_width the width of the big rectangle
    @param small_x the x-coordinate of a corner of a small rectangle
    @param small_y the y-coordinate of ta corner of a small rectangle
    @return true if the small rectangle's corner is inside the big rectangle; false otherwise
    '''
    
#     print("small corner: ", small_x, small_y)
#     print(small_x, "is less than", big_x + big_length, "or ", small_x <= big_x + big_length )
    return small_x >= big_x and small_x <= big_x + big_length and small_y <= big_y and small_y >= big_y - big_width