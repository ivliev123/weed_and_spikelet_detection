import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm

# === НАСТРОЙКИ ===
SOURCE_DIR = "DATASET_WITH_WEED_ALL"  # ваша исходная папка с XML и изображениями
OUTPUT_DIR = "dataset_yolo"           # итоговая структура для YOLO
SPLIT_RATIOS = (0.7, 0.2, 0.1)        # train / val / test

# Классы, которые есть в вашем датасете
CLASSES = ["weed", "spikelet"]  # явно указываем классы для стабильности

def safe_float(value):
    """Безопасное преобразование в float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

def voc_to_yolo_bbox(bbox, img_w, img_h):
    """
    VOC bbox: (xmin, ymin, xmax, ymax)
    YOLO bbox: (x_center, y_center, width, height), нормализованные
    """
    xmin, ymin, xmax, ymax = bbox
    x_c = ((xmin + xmax) / 2) / float(img_w)
    y_c = ((ymin + ymax) / 2) / float(img_h)
    w = (xmax - xmin) / float(img_w)
    h = (ymax - ymin) / float(img_h)
    
    # Проверка на корректность
    if x_c < 0 or x_c > 1 or y_c < 0 or y_c > 1 or w <= 0 or h <= 0:
        print(f"⚠️ Некорректный bbox: {bbox} -> {[x_c, y_c, w, h]}")
        return None
    
    return [x_c, y_c, w, h]

def convert_voc_to_yolo():
    """
    Конвертирует все XML аннотации в формат YOLO
    """
    # Создаем временную папку для всех данных
    temp_images_dir = Path(OUTPUT_DIR) / "temp_images"
    temp_labels_dir = Path(OUTPUT_DIR) / "temp_labels"
    temp_images_dir.mkdir(parents=True, exist_ok=True)
    temp_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Собираем все XML файлы
    source_path = Path(SOURCE_DIR)
    xml_files = list(source_path.rglob("*.xml"))
    print(f"Найдено XML файлов: {len(xml_files)}")
    
    successful_conversions = 0
    all_files = []  # список для хранения путей к файлам для разделения
    
    for xml_path in tqdm(xml_files, desc="Конвертация VOC → YOLO"):
        try:
            # Парсим XML
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Получаем имя файла изображения
            filename_elem = root.find("filename")
            if filename_elem is None:
                print(f"⚠️ Не найден filename в {xml_path}")
                continue
                
            filename = filename_elem.text
            if not filename:
                print(f"⚠️ Пустое filename в {xml_path}")
                continue
            
            # Ищем изображение
            img_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            img_path = None
            for ext in img_extensions:
                # Сначала проверяем как есть
                possible_path = xml_path.parent / filename
                if possible_path.exists():
                    img_path = possible_path
                    break
                
                # Проверяем с заменой расширения
                possible_path = xml_path.parent / (Path(filename).stem + ext)
                if possible_path.exists():
                    img_path = possible_path
                    break
            
            if not img_path or not img_path.exists():
                print(f"⚠️ Не найдено изображение для {xml_path}")
                continue
            
            # Получаем размеры изображения
            size = root.find("size")
            img_w = None
            img_h = None
            
            if size is not None:
                width_elem = size.find("width")
                height_elem = size.find("height")
                
                if width_elem is not None and height_elem is not None:
                    img_w = safe_float(width_elem.text)
                    img_h = safe_float(height_elem.text)
            
            # Если не удалось получить размеры из XML, пробуем получить из изображения
            if img_w is None or img_h is None or img_w <= 0 or img_h <= 0:
                try:
                    # Используем PIL для получения размеров
                    try:
                        from PIL import Image
                        img = Image.open(str(img_path))
                        img_w, img_h = img.size
                        print(f"ℹ️ Размеры из PIL для {xml_path.stem}: {img_w}x{img_h}")
                    except ImportError:
                        # Если PIL нет, пробуем OpenCV
                        import cv2
                        img = cv2.imread(str(img_path))
                        if img is None:
                            print(f"⚠️ Не удалось прочитать изображение: {img_path}")
                            continue
                        img_h, img_w = img.shape[:2]
                        print(f"ℹ️ Размеры из OpenCV для {xml_path.stem}: {img_w}x{img_h}")
                except Exception as e:
                    print(f"⚠️ Не удалось определить размеры изображения {img_path}: {e}")
                    continue
            
            # Проверяем, что размеры валидны
            if img_w <= 0 or img_h <= 0:
                print(f"⚠️ Некорректные размеры изображения: {img_w}x{img_h} в {xml_path}")
                continue
            
            # Создаем YOLO аннотацию
            label_path = temp_labels_dir / f"{xml_path.stem}.txt"
            objects_found = 0
            
            with open(label_path, "w", encoding="utf-8") as f:
                for obj in root.findall("object"):
                    name_elem = obj.find("name")
                    if name_elem is None:
                        continue
                    
                    class_name = name_elem.text.strip() if name_elem.text else ""
                    
                    if not class_name or class_name not in CLASSES:
                        print(f"⚠️ Неизвестный или пустой класс '{class_name}' в {xml_path}")
                        continue
                    
                    class_id = CLASSES.index(class_name)
                    
                    bndbox = obj.find("bndbox")
                    if bndbox is None:
                        continue
                    
                    try:
                        # Безопасное извлечение координат
                        xmin_elem = bndbox.find("xmin")
                        ymin_elem = bndbox.find("ymin")
                        xmax_elem = bndbox.find("xmax")
                        ymax_elem = bndbox.find("ymax")
                        
                        if all(elem is not None for elem in [xmin_elem, ymin_elem, xmax_elem, ymax_elem]):
                            bbox = [
                                safe_float(xmin_elem.text),
                                safe_float(ymin_elem.text),
                                safe_float(xmax_elem.text),
                                safe_float(ymax_elem.text)
                            ]
                            
                            # Проверяем координаты
                            if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                                print(f"⚠️ Некорректные координаты bbox в {xml_path}: {bbox}")
                                continue
                            
                            yolo_bbox = voc_to_yolo_bbox(bbox, img_w, img_h)
                            if yolo_bbox:
                                f.write(f"{class_id} {' '.join(map(lambda x: f'{x:.6f}', yolo_bbox))}\n")
                                objects_found += 1
                    except Exception as e:
                        print(f"⚠️ Ошибка парсинга bbox в {xml_path}: {e}")
                        continue
            
            if objects_found > 0:
                # Копируем изображение
                img_dest = temp_images_dir / img_path.name
                shutil.copy2(img_path, img_dest)
                
                # Сохраняем путь для последующего разделения
                all_files.append({
                    'image': img_dest,
                    'label': label_path,
                    'stem': xml_path.stem
                })
                successful_conversions += 1
            else:
                # Удаляем пустой файл аннотаций
                if label_path.exists():
                    label_path.unlink()
                print(f"⚠️ Нет объектов в {xml_path}")
                
        except Exception as e:
            print(f"❌ Ошибка обработки {xml_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✅ Успешно сконвертировано: {successful_conversions} файлов")
    return all_files

def split_dataset(all_files):
    """
    Разделяет датасет на train/val/test
    """
    if not all_files:
        print("❌ Нет файлов для разделения!")
        return [], [], []
    
    # Создаем финальную структуру папок
    output_path = Path(OUTPUT_DIR)
    for subset in ["train", "val", "test"]:
        (output_path / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / subset).mkdir(parents=True, exist_ok=True)
    
    # Разделяем данные
    train_files, testval_files = train_test_split(
        all_files, 
        test_size=(SPLIT_RATIOS[1] + SPLIT_RATIOS[2]), 
        random_state=42
    )
    
    val_files, test_files = train_test_split(
        testval_files,
        test_size=SPLIT_RATIOS[2] / (SPLIT_RATIOS[1] + SPLIT_RATIOS[2]),
        random_state=42
    )
    
    print(f"\n📊 Распределение данных:")
    print(f"  Train: {len(train_files)} файлов")
    print(f"  Val:   {len(val_files)} файлов")
    print(f"  Test:  {len(test_files)} файлов")
    
    # Копируем файлы в соответствующие папки
    def copy_files(files, subset):
        for file_info in files:
            # Копируем изображение
            img_src = file_info['image']
            img_dst = output_path / "images" / subset / img_src.name
            shutil.copy2(img_src, img_dst)
            
            # Копируем аннотацию
            label_src = file_info['label']
            label_dst = output_path / "labels" / subset / f"{file_info['stem']}.txt"
            shutil.copy2(label_src, label_dst)
    
    copy_files(train_files, "train")
    copy_files(val_files, "val")
    copy_files(test_files, "test")
    
    # Удаляем временные папки
    temp_images_path = output_path / "temp_images"
    temp_labels_path = output_path / "temp_labels"
    
    if temp_images_path.exists():
        shutil.rmtree(temp_images_path)
    if temp_labels_path.exists():
        shutil.rmtree(temp_labels_path)
    
    return train_files, val_files, test_files

def create_yaml():
    """
    Создает dataset.yaml файл для YOLO
    """
    yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
    
    # Получаем абсолютный путь к датасету
    dataset_path = Path(OUTPUT_DIR).absolute()
    
    yaml_content = f"""# YOLO Dataset Configuration
# Generated automatically

path: {dataset_path}  # dataset root dir
train: images/train    # train images
val: images/val        # val images
test: images/test      # test images

# Classes
nc: {len(CLASSES)}  # number of classes
names: {CLASSES}    # class names

# Download script/API (optional)
# Example:
# download: |
#   from utils.general import download
#   # Download labels
"""

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print(f"\n✅ Создан YAML файл: {yaml_path}")
    return yaml_path

def analyze_dataset():
    """
    Анализирует итоговый датасет
    """
    print("\n📈 Анализ датасета:")
    
    output_path = Path(OUTPUT_DIR)
    for subset in ["train", "val", "test"]:
        images_dir = output_path / "images" / subset
        labels_dir = output_path / "labels" / subset
        
        if images_dir.exists():
            num_images = len([f for f in images_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            num_labels = len([f for f in labels_dir.iterdir() if f.suffix == '.txt'])
            
            print(f"  {subset.capitalize():5s}: {num_images:4d} изображений, {num_labels:4d} аннотаций")
            
            # Подсчет объектов по классам
            if num_labels > 0:
                class_counts = {i: 0 for i in range(len(CLASSES))}
                for label_file in labels_dir.iterdir():
                    if label_file.suffix == '.txt':
                        with open(label_file, 'r') as f:
                            lines = f.readlines()
                            for line in lines:
                                if line.strip():
                                    try:
                                        class_id = int(line.split()[0])
                                        if class_id in class_counts:
                                            class_counts[class_id] += 1
                                    except:
                                        continue
                
                print(f"        Объекты: " + ", ".join([f"{CLASSES[i]}: {count}" for i, count in class_counts.items()]))

def main():
    """
    Основная функция для конвертации и разделения датасета
    """
    print("🚀 Начало конвертации датасета...")
    print(f"Исходная папка: {SOURCE_DIR}")
    print(f"Целевая папка: {OUTPUT_DIR}")
    print(f"Классы: {CLASSES}")
    
    # Убедимся, что выходная директория существует
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    # Шаг 1: Конвертируем VOC в YOLO
    print("\n" + "="*50)
    print("ШАГ 1: Конвертация VOC XML → YOLO формат")
    print("="*50)
    all_files = convert_voc_to_yolo()
    
    if not all_files:
        print("❌ Не найдено файлов для конвертации!")
        return
    
    # Шаг 2: Разделяем на train/val/test
    print("\n" + "="*50)
    print("ШАГ 2: Разделение датасета")
    print("="*50)
    train_files, val_files, test_files = split_dataset(all_files)
    
    # Шаг 3: Создаем YAML файл
    print("\n" + "="*50)
    print("ШАГ 3: Создание конфигурационного файла")
    print("="*50)
    yaml_path = create_yaml()
    
    # Шаг 4: Анализ
    print("\n" + "="*50)
    print("ШАГ 4: Анализ итогового датасета")
    print("="*50)
    analyze_dataset()
    
    # Итог
    print("\n" + "="*50)
    print("✅ КОНВЕРТАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
    print("="*50)
    print(f"\nСтруктура датасета:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── dataset.yaml")
    print(f"  ├── images/")
    print(f"  │   ├── train/")
    print(f"  │   ├── val/")
    print(f"  │   └── test/")
    print(f"  └── labels/")
    print(f"      ├── train/")
    print(f"      ├── val/")
    print(f"      └── test/")
    
    print(f"\nИспользование в YOLO:")
    print(f"  model.train(data='{yaml_path}', epochs=100, imgsz=800)")

if __name__ == "__main__":
    # Сначала установите необходимые библиотеки если их нет
    try:
        from PIL import Image
    except ImportError:
        print("Установите PIL: pip install pillow")
    
    try:
        from tqdm import tqdm
    except ImportError:
        print("Установите tqdm: pip install tqdm")
    
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("Установите scikit-learn: pip install scikit-learn")
    
    main()