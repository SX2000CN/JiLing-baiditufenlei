import os
import shutil
import zipfile
from datetime import datetime

def create_training_package():
    """创建训练工具包"""
    
    # 基础路径
    base_path = r"D:\桌面\编程\服装挂拍分类\训练"
    package_name = f"服装分类训练工具_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    package_dir = os.path.join(base_path, package_name)
    
    print(f"开始创建训练工具包: {package_name}")
    
    # 创建包目录
    os.makedirs(package_dir, exist_ok=True)
    
    # 1. 复制训练脚本
    scripts_to_copy = [
        "train_efficientnetb0_448_2.py",
        "train_efficientnetb0_512.py"
    ]
    
    scripts_dir = os.path.join(package_dir, "训练脚本")
    os.makedirs(scripts_dir, exist_ok=True)
    
    for script in scripts_to_copy:
        src_path = os.path.join(base_path, script)
        if os.path.exists(src_path):
            dst_path = os.path.join(scripts_dir, script)
            shutil.copy2(src_path, dst_path)
            print(f"已复制训练脚本: {script}")
    
    # 2. 创建必要的文件夹结构
    folders_to_create = [
        "模型装载",
        "筛选",
        "新模型"
    ]
    
    for folder in folders_to_create:
        folder_path = os.path.join(package_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        # 在每个文件夹中创建说明文件
        readme_path = os.path.join(folder_path, "README.txt")
        with open(readme_path, 'w', encoding='utf-8') as f:
            if folder == "模型装载":
                f.write("此文件夹用于存放预训练模型文件\n")
                f.write("支持的格式: .keras, .h5\n")
                f.write("训练程序会自动加载此文件夹中最新的模型文件\n")
            elif folder == "筛选":
                f.write("此文件夹用于存放训练数据\n")
                f.write("请按照以下结构组织数据:\n")
                f.write("筛选/\n")
                f.write("  ├── 主图/\n")
                f.write("  ├── 吊牌/\n")
                f.write("  └── 细节/\n")
                f.write("\n每个子文件夹存放对应类别的图片")
            elif folder == "新模型":
                f.write("此文件夹用于保存训练完成的新模型\n")
                f.write("训练过程中会自动保存:\n")
                f.write("- 最佳模型检查点\n")
                f.write("- 最终训练模型\n")
                f.write("- 类别映射文件 (class_names.json)\n")
        
        print(f"已创建文件夹: {folder}")
    
    # 3. 复制现有的模型文件(如果存在)
    src_model_dir = os.path.join(base_path, "模型装载")
    dst_model_dir = os.path.join(package_dir, "模型装载")
    
    if os.path.exists(src_model_dir):
        model_files = [f for f in os.listdir(src_model_dir) 
                      if f.lower().endswith(('.keras', '.h5', '.json'))]
        for model_file in model_files:
            src_file = os.path.join(src_model_dir, model_file)
            dst_file = os.path.join(dst_model_dir, model_file)
            shutil.copy2(src_file, dst_file)
            print(f"已复制模型文件: {model_file}")
    
    # 4. 复制训练数据样本(如果存在且不太大)
    src_data_dir = os.path.join(base_path, "筛选")
    dst_data_dir = os.path.join(package_dir, "筛选")
    
    if os.path.exists(src_data_dir):
        try:
            # 计算数据大小
            total_size = 0
            for root, dirs, files in os.walk(src_data_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.exists(file_path):
                        total_size += os.path.getsize(file_path)
            
            # 如果数据小于500MB则复制，否则只复制文件夹结构
            if total_size < 500 * 1024 * 1024:  # 500MB
                for item in os.listdir(src_data_dir):
                    if os.path.isdir(os.path.join(src_data_dir, item)):
                        src_subdir = os.path.join(src_data_dir, item)
                        dst_subdir = os.path.join(dst_data_dir, item)
                        shutil.copytree(src_subdir, dst_subdir, dirs_exist_ok=True)
                        print(f"已复制训练数据: {item}")
            else:
                print(f"训练数据过大({total_size/1024/1024:.1f}MB)，仅复制文件夹结构")
                # 只创建子文件夹结构
                for item in os.listdir(src_data_dir):
                    if os.path.isdir(os.path.join(src_data_dir, item)):
                        os.makedirs(os.path.join(dst_data_dir, item), exist_ok=True)
                        
        except Exception as e:
            print(f"复制训练数据时出错: {e}")
    
    # 5. 创建使用说明文档
    usage_doc = os.path.join(package_dir, "使用说明.txt")
    with open(usage_doc, 'w', encoding='utf-8') as f:
        f.write("服装分类训练工具使用说明\n")
        f.write("=" * 40 + "\n\n")
        f.write("1. 环境要求:\n")
        f.write("   - Python 3.8+\n")
        f.write("   - TensorFlow 2.x\n")
        f.write("   - 其他依赖: numpy, pillow\n\n")
        f.write("2. 使用步骤:\n")
        f.write("   a) 准备训练数据:\n")
        f.write("      将图片按类别放入 '筛选' 文件夹的对应子文件夹中\n\n")
        f.write("   b) (可选) 预训练模型:\n")
        f.write("      将预训练模型文件放入 '模型装载' 文件夹\n\n")
        f.write("   c) 运行训练:\n")
        f.write("      python train_efficientnetb0_448_2.py  # 标准训练\n")
        f.write("      python train_efficientnetb0_512.py   # 512尺寸训练\n\n")
        f.write("   d) 获取结果:\n")
        f.write("      训练完成的模型会保存在 '新模型' 文件夹中\n\n")
        f.write("3. 文件夹说明:\n")
        f.write("   - 训练脚本/: 包含训练程序\n")
        f.write("   - 模型装载/: 存放预训练模型\n")
        f.write("   - 筛选/: 存放训练数据\n")
        f.write("   - 新模型/: 保存训练结果\n\n")
        f.write("4. 注意事项:\n")
        f.write("   - 确保显卡驱动和CUDA环境正确安装\n")
        f.write("   - 训练过程中请勿关闭程序\n")
        f.write("   - 建议定期备份重要模型文件\n")
    
    # 6. 创建依赖安装脚本
    install_script = os.path.join(package_dir, "install_requirements.bat")
    with open(install_script, 'w', encoding='utf-8') as f:
        f.write("@echo off\n")
        f.write("echo 正在安装训练工具依赖...\n")
        f.write("pip install tensorflow>=2.8.0\n")
        f.write("pip install pillow\n")
        f.write("pip install numpy\n")
        f.write("pip install matplotlib\n")
        f.write("echo 依赖安装完成!\n")
        f.write("pause\n")
    
    # 7. 创建快速启动脚本
    start_script = os.path.join(package_dir, "开始训练.bat")
    with open(start_script, 'w', encoding='utf-8') as f:
        f.write("@echo off\n")
        f.write("cd /d %~dp0\n")
        f.write("echo 服装分类训练工具\n")
        f.write("echo ==================\n")
        f.write("echo 1. 标准训练 (train_efficientnetb0_448_2.py)\n")
        f.write("echo 2. 512尺寸训练 (train_efficientnetb0_512.py)\n")
        f.write("set /p choice=请选择训练模式 (1 或 2): \n")
        f.write("if \"%choice%\"==\"1\" (\n")
        f.write("    python 训练脚本\\train_efficientnetb0_448_2.py\n")
        f.write(") else if \"%choice%\"==\"2\" (\n")
        f.write("    python 训练脚本\\train_efficientnetb0_512.py\n")
        f.write(") else (\n")
        f.write("    echo 无效选择\n")
        f.write(")\n")
        f.write("pause\n")
    
    print(f"\n训练工具包创建完成!")
    print(f"位置: {package_dir}")
    print(f"大小: {get_folder_size(package_dir):.1f} MB")
    
    # 询问是否创建ZIP压缩包
    create_zip = input("\n是否创建ZIP压缩包? (y/n): ").lower().strip()
    if create_zip in ['y', 'yes', '是']:
        zip_path = f"{package_dir}.zip"
        create_zip_package(package_dir, zip_path)
        print(f"ZIP压缩包已创建: {zip_path}")
    
    return package_dir

def get_folder_size(folder_path):
    """计算文件夹大小(MB)"""
    total_size = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size / (1024 * 1024)

def create_zip_package(source_dir, zip_path):
    """创建ZIP压缩包"""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(source_dir))
                zipf.write(file_path, arcname)

if __name__ == "__main__":
    try:
        package_dir = create_training_package()
        print("\n打包完成!")
    except Exception as e:
        print(f"打包过程中出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("按回车键退出...")
