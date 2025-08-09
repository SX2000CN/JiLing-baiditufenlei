# -*- mode: python -*-
import os
import shutil
import sys
import numpy as np
import json
import time

# —— 强制 PyInstaller 打包 tensorflow.keras 相关模块 —— 
import tensorflow.keras.models
import tensorflow.keras.preprocessing.image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def main():
    start_time = time.time()
    
    # ===== 配置区 =====
    base_path = getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))
    model_dir = os.path.join(base_path, '模型', '装载')
    input_dir = r'D:\桌面\筛选\JPG'
    
    # 动态加载类别名称，如果不存在则使用默认值
    class_names_file = os.path.join(model_dir, 'class_names.json')
    if os.path.exists(class_names_file):
        try:
            with open(class_names_file, 'r', encoding='utf-8') as f:
                class_names = json.load(f)
            print(f"从文件加载类别名称: {class_names}")
        except Exception as e:
            print(f"加载类别文件失败: {e}，使用默认类别")
            class_names = ['主图', '吊牌', '细节']
    else:
        print("未找到类别文件，使用默认类别")
        class_names = ['主图', '吊牌', '细节']
    
    # 动态构建输出目录
    output_dirs = {}
    for class_name in class_names:
        output_dirs[class_name] = rf'D:\桌面\筛选\{class_name}'
    
    img_height, img_width = 512, 512
    batch_size = 48
    confidence_threshold = 0.5  # 添加置信度阈值
    
    print(f"配置信息:")
    print(f"- 输入目录: {input_dir}")
    print(f"- 图像尺寸: {img_width}x{img_height}")
    print(f"- 批处理大小: {batch_size}")
    print(f"- 置信度阈值: {confidence_threshold}")

    # ===== 自动查找模型文件并加载 =====
    def find_model_file(model_dir):
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"模型目录不存在: {model_dir}")
        files = [f for f in os.listdir(model_dir) if f.lower().endswith(('.keras', '.h5'))]
        if not files:
            raise FileNotFoundError(f"模型文件未在 {model_dir} 下找到！")
        files = sorted(
            files,
            key=lambda f: os.path.getmtime(os.path.join(model_dir, f)),
            reverse=True
        )
        model_path = os.path.join(model_dir, files[0])
        print(f"自动加载模型: {model_path}")
        return model_path

    try:
        model_path = find_model_file(model_dir)
        model = load_model(model_path)
        
        # 验证模型输出维度
        expected_classes = len(class_names)
        actual_classes = model.output_shape[-1]
        if actual_classes != expected_classes:
            print(f"警告：模型输出维度({actual_classes})与类别数({expected_classes})不匹配！")
            if actual_classes < expected_classes:
                class_names = class_names[:actual_classes]
                print(f"已调整类别列表为: {class_names}")
        
        print(f"模型加载成功，类别数: {len(class_names)}")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # ===== 检查输入目录 =====
    if not os.path.exists(input_dir):
        print(f"输入目录不存在: {input_dir}")
        return

    # ===== 获取所有图片文件 =====
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(img_exts)]
    total = len(files)
    if total == 0:
        print("没有检测到图片文件。")
        return
    print(f"共检测到 {total} 张图片。开始批量分类...")

    # 创建输出目录
    for out_dir in output_dirs.values():
        os.makedirs(out_dir, exist_ok=True)
    
    # ===== 批量分类 =====
    processed = 0
    successful = 0
    failed = 0
    low_confidence_count = 0
    category_counts = {name: 0 for name in class_names}  # 新增：各类别计数
    
    processing_start = time.time()
    
    for i in range(0, total, batch_size):
        batch_start = time.time()
        batch_files = files[i:i+batch_size]
        batch_imgs = []
        valid_files = []
        
        print(f"处理批次 {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch_files)} 张图片)")
        
        for file in batch_files:
            img_path = os.path.join(input_dir, file)
            try:
                img = image.load_img(img_path, target_size=(img_height, img_width))
                img_array = image.img_to_array(img) / 255.0
                batch_imgs.append(img_array)
                valid_files.append(file)
            except Exception as e:
                print(f"加载图片 {file} 时出错：{e}")
                failed += 1
                
        if not batch_imgs:
            continue
            
        try:
            batch_imgs_np = np.array(batch_imgs)
            preds = model.predict(batch_imgs_np, verbose=0)
            
            for j, pred in enumerate(preds):
                file = valid_files[j]
                img_path = os.path.join(input_dir, file)
                processed += 1
                
                pred_class = np.argmax(pred)
                max_confidence = np.max(pred)
                pred_label = class_names[pred_class]
                
                # 检查置信度
                if max_confidence < confidence_threshold:
                    # 低置信度图片移动到细节文件夹
                    detail_dir = output_dirs.get('细节', r'D:\桌面\筛选\细节')
                    os.makedirs(detail_dir, exist_ok=True)
                    shutil.copy(img_path, os.path.join(detail_dir, file))
                    print(f"{file} → 细节(低置信度:{max_confidence:.3f}) → 原预测:{pred_label}")
                    low_confidence_count += 1
                    category_counts['细节'] += 1  # 计入细节类别
                else:
                    if pred_label in output_dirs:
                        out_dir = output_dirs[pred_label]
                        shutil.copy(img_path, os.path.join(out_dir, file))
                        print(f"{file} → {pred_label} (置信度: {max_confidence:.3f})")
                        successful += 1
                        category_counts[pred_label] += 1  # 新增：计数
                    else:
                        print(f"{file} → 预测为 {pred_label}，不做移动。")
                        failed += 1
                        
        except Exception as e:
            print(f"批次预测失败: {e}")
            failed += len(valid_files)
        
        batch_time = time.time() - batch_start
        print(f"批次处理耗时: {batch_time:.2f}秒")

    processing_time = time.time() - processing_start
    
    print(f"\n=== 分类完成 ===")
    print(f"总处理: {processed} 张")
    print(f"成功分类: {successful} 张")
    print(f"低置信度(归入细节): {low_confidence_count} 张")
    print(f"失败: {failed} 张")
    print(f"处理耗时: {processing_time:.2f}秒")
    print(f"平均速度: {processed/processing_time:.1f} 张/秒")
    
    # 新增：各类别统计
    print(f"\n=== 各类别统计 ===")
    for class_name, count in category_counts.items():
        print(f"{class_name}: {count} 张")

    # ===== 删除JPG文件夹中的所有图片 =====
    if successful > 0 or low_confidence_count > 0:
        print("\n开始清空原始文件夹...")
        deleted_count = 0
        for file in files:
            img_path = os.path.join(input_dir, file)
            try:
                os.remove(img_path)
                deleted_count += 1
            except Exception as e:
                print(f"删除 {file} 时出错：{e}")
        print(f"已删除 {deleted_count} 个文件，JPG文件夹已清空。")

    total_time = time.time() - start_time
    print(f"程序总耗时: {total_time:.2f}秒")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"程序发生错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        input("按回车键退出...")  # 方便exe
        input("按回车键退出...")  # 方便exe
