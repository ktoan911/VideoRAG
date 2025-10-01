
metadata_db = [
    {
        "video_id": "program_A_2024",
        "scene_id": "scene_001",
        "timestamp": (0, 65), # (start_sec, end_sec)
        "blended_text": "MC giới thiệu về mùa hoa anh đào ở Kyoto. Cảnh quay flycam tuyệt đẹp trên bầu trời thành phố, bao phủ bởi sắc hồng của hoa.",
        "keyframe_embedding": [0.1, 0.4, ..., 0.9], # Vector của keyframe chính
        "text_embedding": [0.2, 0.5, ..., 0.8], # Vector của blended_text
        "persons_in_scene": [], # Không có ai được nhận dạng
        "transcript": "Xin chào quý vị khán giả, hôm nay chúng ta sẽ đến với thành phố Kyoto..."
    },
    {
        "video_id": "talkshow_B_2024",
        "scene_id": "scene_002",
        "timestamp": (120, 180),
        "blended_text": "Diễn viên Tanaka Ken kể về kỷ niệm thời thơ ấu của mình. Anh ấy nhớ lại những ngày tháng chơi đùa ở quê nhà. Khuôn mặt anh ấy biểu lộ cảm xúc sâu sắc.",
        "keyframe_embedding": [0.5, 0.3, ..., 0.1],
        "text_embedding": [0.6, 0.2, ..., 0.3],
        "persons_in_scene": ["Tanaka Ken"],
        "transcript": "Tanaka Ken: (Cười) Hồi nhỏ, tôi hay cùng bạn bè đi tắm sông..."
    },
    # ... và nhiều cảnh khác
]