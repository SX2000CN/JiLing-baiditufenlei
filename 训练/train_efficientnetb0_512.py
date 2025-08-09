import os
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras import callbacks
import json

# 1. 路径参数 - 修改为相对路径以支持打包
base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
base_dir = os.path.join(base_path, "筛选")
model_dir = os.path.join(base_path, "模型装载")
save_dir = os.path.join(base_path, "新模型")

img_height, img_width = 512, 512
batch_size = 48

print(f"工作目录: {base_path}")
print(f"训练数据目录: {base_dir}")
print(f"模型装载目录: {model_dir}")
print(f"模型保存目录: {save_dir}")

# 2. 优化数据增强 - 分离训练和验证
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,  # 减少旋转角度
    width_shift_range=0.15,  # 减少平移范围
    height_shift_range=0.15,
    zoom_range=0.15,  # 减少缩放范围
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # 减少亮度变化
    shear_range=0.1,  # 添加剪切变换
    fill_mode='nearest',
    validation_split=0.2
)

# 验证集只做归一化
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=42
)
val_generator = val_datagen.flow_from_directory(
    base_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False,  # 验证集不打乱
    seed=42
)

class_indices = train_generator.class_indices
class_names = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name
num_classes = len(class_names)
print(f"Detected classes ({num_classes}): {class_names}")

# 新增：保存类别顺序到文件，供预测脚本读取
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, "class_names.json"), "w", encoding="utf-8") as jf:
    json.dump(class_names, jf, ensure_ascii=False)

# 新增：数据验证
print(f"训练集样本数: {train_generator.samples}")
print(f"验证集样本数: {val_generator.samples}")
print(f"每轮训练步数: {train_generator.samples // batch_size}")
print(f"每轮验证步数: {val_generator.samples // batch_size}")

# 3. 自动查找模型装载文件夹下最新的模型
def find_model_file(model_dir):
    if not os.path.exists(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return None
    files = [f for f in os.listdir(model_dir) if f.lower().endswith(('.keras', '.h5'))]
    if not files:
        return None
    files = sorted(files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
    return os.path.join(model_dir, files[0])

model_path = find_model_file(model_dir)

# 改进模型加载逻辑
if model_path:
    print(f"自动加载模型: {model_path}")
    try:
        model = load_model(model_path)
        print(f"成功加载模型，输出类别数: {model.output_shape[-1]}")
        
        # 验证模型输出维度是否匹配当前数据集
        if model.output_shape[-1] != num_classes:
            print(f"警告：模型输出维度({model.output_shape[-1]})与当前数据集类别数({num_classes})不匹配！")
            print("将重新构建模型...")
            model_path = None  # 强制重新构建
        else:
            # 尝试获取base_model，如果失败则继续训练整个模型
            try:
                base_model = None
                for layer in model.layers:
                    if hasattr(layer, 'name') and 'efficientnet' in layer.name.lower():
                        base_model = layer
                        break
                if base_model is None:
                    print("未找到EfficientNet基础层，将训练整个模型")
            except:
                print("无法识别模型结构，将训练整个模型")
                base_model = None
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("将从头训练新模型")
        model_path = None

if not model_path:
    print(f"构建新的EfficientNetB0模型...")
    from tensorflow.keras.applications import EfficientNetB0
    base_model = EfficientNetB0(
        include_top=False, weights='imagenet',
        input_shape=(img_height, img_width, 3)
    )
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = base_model(inputs, training=False)  # 初始冻结
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)  # 添加Dropout
    x = keras.layers.Dense(256, activation='relu')(x)  # 添加中间层
    x = keras.layers.BatchNormalization()(x)  # 批归一化
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    print(f"新模型构建完成，类别数: {num_classes}")

# 新增：显示模型摘要
print("\n模型结构摘要:")
print(f"总参数数: {model.count_params():,}")
trainable_params = sum([tf.size(var).numpy() for var in model.trainable_variables])
print(f"可训练参数数: {trainable_params:,}")
print(f"输入形状: {model.input_shape}")
print(f"输出形状: {model.output_shape}")

# 添加训练进度回调
class TrainingProgress(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\n=== 开始第 {epoch + 1} 轮训练 ===")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"第 {epoch + 1} 轮完成 - "
              f"损失: {logs.get('loss', 0):.4f}, "
              f"准确率: {logs.get('accuracy', 0):.4f}, "
              f"验证损失: {logs.get('val_loss', 0):.4f}, "
              f"验证准确率: {logs.get('val_accuracy', 0):.4f}")

# 4. 分阶段训练策略
def train_model_staged():
    progress_callback = TrainingProgress()
    
    # 阶段1：冻结基础模型，只训练分类头
    if base_model:
        base_model.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("阶段1：训练分类头...")
    history1 = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[progress_callback],
        verbose=1
    )
    
    # 阶段2：解冻部分层进行精调
    if base_model:
        base_model.trainable = True
        # 冻结前面的层，只训练后面的层
        for layer in base_model.layers[:-30]:
            layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("阶段2：部分精调...")
    history2 = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator,
        callbacks=[early_stop, checkpoint, progress_callback],
        verbose=1
    )
    
    # 阶段3：全网络精调
    if base_model:
        for layer in base_model.layers:
            layer.trainable = True
    
    # 使用学习率调度
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("阶段3：全网络精调...")
    history3 = model.fit(
        train_generator,
        epochs=15,
        validation_data=val_generator,
        callbacks=[early_stop, checkpoint, lr_scheduler, progress_callback],
        verbose=1
    )
    
    return history1, history2, history3

# 5. 优化回调设置
os.makedirs(save_dir, exist_ok=True)
checkpoint = callbacks.ModelCheckpoint(
    os.path.join(save_dir, "clothes_classifier_efficientnetb0_448_continue.keras"), 
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=8,  # 增加耐心值
    restore_best_weights=True,
    mode='max',
    verbose=1
)

# 6. 执行分阶段训练
if model_path and base_model:
    # 如果加载了已有模型，直接进行精调
    if base_model:
        base_model.trainable = True
    
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    
    progress_callback = TrainingProgress()
    
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("继续训练已有模型...")
    history = model.fit(
        train_generator,
        epochs=20,  # 增加训练轮数
        validation_data=val_generator,
        callbacks=[early_stop, checkpoint, lr_scheduler, progress_callback]
    )
else:
    # 新模型使用分阶段训练
    history1, history2, history3 = train_model_staged()

# 7. 保存最终模型到新模型目录
final_model_path = os.path.join(save_dir, "clothes_classifier_efficientnetb0_448_continue_final.keras")
model.save(final_model_path)
print(f"最终模型已保存到: {final_model_path}")