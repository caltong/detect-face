# detect-face
对简历扫描件进行裁切 获取人物头像照片
# 环境要求
详见environment.yaml
# 流程
1. cut_photo.py 使用opencv识别人脸初步估计照片位置
2. cut_photo_use_model.py 使用训练获得的模型裁切背景
3. resize_use_mtcnn.py 使用mtcnn识别人脸关键点 进行大小尺寸裁切
# 背景裁切模型训练
1. model_use_vgg16_fine_tuning.py 使用vgg16 fine tuning
2. model_step_by_step.py 多次生成数据继续训练
# 数据
1. data_augmentation.py 随机生成数据
2. load_data.py 读取数据 