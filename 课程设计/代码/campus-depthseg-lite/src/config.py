"""Small central configuration for Campus DepthSeg Lite."""

CLASS_NAMES = ["other", "floor", "wall", "obstacle", "door_window"]
IGNORE_INDEX = 255
NUM_CLASSES = len(CLASS_NAMES)

# NYU40 class ids are mapped into the five course-design classes.
# 0 is kept as "other" so tiny synthetic labels can also pass through tests.
NYU40_TO_5 = {
    0: 0,
    1: 2,   # wall
    2: 1,   # floor
    3: 3,   # cabinet
    4: 3,   # bed
    5: 3,   # chair
    6: 3,   # sofa
    7: 3,   # table
    8: 4,   # door
    9: 4,   # window
    10: 3,  # bookshelf
    11: 0,  # picture
    12: 3,  # counter
    13: 4,  # blinds
    14: 3,  # desk
    15: 3,  # shelves
    16: 4,  # curtain
    17: 3,  # dresser
    18: 3,  # pillow
    19: 0,  # mirror
    20: 3,  # floor mat
    21: 3,  # clothes
    22: 0,  # ceiling
    23: 3,  # books
    24: 3,  # refrigerator
    25: 3,  # television
    26: 3,  # paper
    27: 3,  # towel
    28: 4,  # shower curtain
    29: 3,  # box
    30: 0,  # whiteboard
    31: 3,  # person
    32: 3,  # night stand
    33: 3,  # toilet
    34: 3,  # sink
    35: 3,  # lamp
    36: 3,  # bathtub
    37: 3,  # bag
    38: 0,  # other structure
    39: 3,  # other furniture
    40: 3,  # other prop
}

RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)

IMAGE_SIZE = (240, 320)
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0
DICE_WEIGHT = 1.0
