from db import register_video

videos = [ # Videos stored as a list of dicts — easy to add new videos later, just add a new {} block.
    {
        'file_path':    'videos/IMG_0350.MOV',
        'session_type': 'tagging',
        'obstacles':    True,
        'fish_count':   5,
        'notes':        "None"
    },
    {
        'file_path':    'videos/IMG_0651.MOV',
        'session_type': 'tagging',
        'obstacles':    True,
        'fish_count':   5,
        'notes':        "useful to label reflection"
    },
    {
        'file_path':    'videos/IMG_9856.MOV',
        'session_type': 'tagging',
        'obstacles':    True,
        'fish_count':   1,
        'notes':        "has also one dead fish"
    },
    {
        'file_path':    'videos/IMG_0764.MOV',
        'session_type': 'tagging',
        'obstacles':    False,
        'fish_count':   1,
        'notes':        "has also one dead fish, could to focus on reflection"
    }
]

for v in videos:
    video_id = register_video(**v) #  unpacks the dict into function parameters /unpacks dict into separate named arguments / same as same as: register_video(file_path='videos/IMG_0350.MOV', session_type='behaviour', ...)
    print(f"Registered {v['file_path']} → video_id={video_id}")
