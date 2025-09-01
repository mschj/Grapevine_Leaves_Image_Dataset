#!/usr/bin/env python
# coding: utf-8

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.applications import DenseNet201
import time
from datetime import datetime
import csv

# 开始时间记录
starttime = time.time()
nowtime1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print("开始时间=" + nowtime1)

# 数据集配置
seed = 123
data_dir = r'D:\DataSet\1 Grapevine Leaves Image/Grapevine_Leaves_Image_Dataset'
image_size = (256, 256)
batch_size = 32
num_classes = 5  # 数据集包含5个类别

# 划分数据集：80%训练集，20%临时集（用于划分验证集和测试集）
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,  # 20%用于验证+测试
    subset='training',
    seed=seed
)

# 20%的临时集
temp_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=image_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed
)

# 从20%临时集中再划分50%作为验证集（总数据的10%），50%作为测试集（总数据的10%）
temp_size = len(temp_ds)
val_size = int(temp_size * 0.5)

# 最终数据集
val_ds = temp_ds.take(val_size)
test_ds = temp_ds.skip(val_size)

# 数据归一化
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

# 打印数据集信息
print(f"训练集批次数量: {len(train_ds)}")
print(f"验证集批次数量: {len(val_ds)}")
print(f"测试集批次数量: {len(test_ds)}")


# 数据可视化
def visualize_data(category):
    path = os.path.join(data_dir, category)
    if not os.path.exists(path):
        print(f"类别路径不存在: {path}")
        return

    image_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    if len(image_files) == 0:
        print(f"类别 {category} 中没有图片文件")
        return

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(min(6, len(image_files))):
        image_file = image_files[i]
        label = image_file.split('.')[0]

        img_path = os.path.join(path, image_file)
        img = mpimg.imread(img_path)
        ax = axs[i // 3, i % 3]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(label)

    plt.tight_layout()
    plt.show()


# 可视化各个类别的样本
categories = ['Ak', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli']
for category in categories:
    visualize_data(category)

# 构建模型
conv_base = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3),
    pooling='avg'
)
conv_base.trainable = False  # 冻结预训练层

model = Sequential()
model.add(conv_base)
model.add(BatchNormalization())
model.add(Dropout(0.5))

# 全连接层
model.add(Dense(256, activation=LeakyReLU(alpha=0.01),
                kernel_initializer='he_normal',
                kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation=LeakyReLU(alpha=0.01),
                kernel_initializer='he_normal',
                kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Dense(64, activation=LeakyReLU(alpha=0.01),
                kernel_initializer='he_normal'))

# 输出层
model.add(Dense(5, activation='softmax'))

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc', multi_label=True)
    ]
)

# 学习率调度器（不使用早停策略）
lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.1,
    patience=3,
    verbose=1,
    mode='max',
    min_lr=0.000001
)

# 创建结果保存目录
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("DenseNet201_results", current_time)
os.makedirs(results_dir, exist_ok=True)


# 保存指标到CSV的工具函数
def save_metrics_to_csv(filename, metrics, is_header=False):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if is_header:
            writer.writerow(metrics)
        else:
            writer.writerow(metrics)


# 初始化CSV文件并写入表头
train_csv = os.path.join(results_dir, 'train_metrics.csv')
val_csv = os.path.join(results_dir, 'val_metrics.csv')
test_csv = os.path.join(results_dir, 'test_metrics.csv')

# 表头
header = ['epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1_score', 'auc']

# 写入表头
save_metrics_to_csv(train_csv, header, is_header=True)
save_metrics_to_csv(val_csv, header, is_header=True)
save_metrics_to_csv(test_csv, header, is_header=True)


# 自定义回调函数，保存训练和验证指标
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # 计算F1分数
        if logs.get('precision') and logs.get('recall'):
            train_f1 = 2 * (logs.get('precision') * logs.get('recall')) / (
                        logs.get('precision') + logs.get('recall') + 1e-7)
        else:
            train_f1 = 0.0

        if logs.get('val_precision') and logs.get('val_recall'):
            val_f1 = 2 * (logs.get('val_precision') * logs.get('val_recall')) / (
                        logs.get('val_precision') + logs.get('val_recall') + 1e-7)
        else:
            val_f1 = 0.0

        # 保存训练指标
        train_metrics = [
            epoch + 1,
            logs.get('loss'),
            logs.get('accuracy'),
            logs.get('precision'),
            logs.get('recall'),
            train_f1,
            logs.get('auc')
        ]
        save_metrics_to_csv(train_csv, train_metrics)

        # 保存验证指标
        val_metrics = [
            epoch + 1,
            logs.get('val_loss'),
            logs.get('val_accuracy'),
            logs.get('val_precision'),
            logs.get('val_recall'),
            val_f1,
            logs.get('val_auc')
        ]
        save_metrics_to_csv(val_csv, val_metrics)


# 生成并保存ROC曲线
def plot_roc_curve(y_true, y_pred, title, filename):
    plt.figure(figsize=(10, 8))

    # 多类别的ROC曲线
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()


# 评估并保存指标（修复了numpy错误）
def evaluate_and_save_metrics(dataset, dataset_name, model, results_dir, final_epoch):
    # 评估数据集
    metrics = model.evaluate(dataset, return_dict=True, verbose=1)

    # 计算F1分数
    y_true = []
    y_pred_probs = []

    for x, y in dataset:
        y_true.extend(y.numpy())  # y是Tensor，需要转为NumPy数组
        # 关键修复：model.predict返回的已经是NumPy数组，无需再调用.numpy()
        y_pred_probs.extend(model.predict(x, verbose=0))

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)
    true_labels = np.argmax(y_true, axis=1)

    # 计算宏观平均F1分数
    f1 = f1_score(true_labels, y_pred, average='macro')

    # 保存最终指标
    csv_file = os.path.join(results_dir, f'{dataset_name}_metrics.csv')
    final_metrics = [
        final_epoch,
        metrics['loss'],
        metrics['accuracy'],
        metrics['precision'],
        metrics['recall'],
        f1,
        metrics['auc']
    ]
    save_metrics_to_csv(csv_file, final_metrics)

    # 绘制并保存ROC曲线
    roc_filename = os.path.join(results_dir, f'{dataset_name}_roc_curve.png')
    plot_roc_curve(y_true, y_pred_probs, f'ROC Curve - {dataset_name.capitalize()}', roc_filename)

    # 将F1分数添加到返回的指标中
    metrics['f1_score'] = f1
    return metrics


# 训练模型（不使用早停，确保训练100个epoch）
history = model.fit(
    train_ds,
    epochs=100,  # 明确指定100个epoch
    validation_data=val_ds,
    callbacks=[MetricsCallback(), lr_scheduler]  # 仅保留学习率调度器
)

# 实际训练轮次固定为100
actual_epochs = 100

# 评估并保存所有数据集的指标
print("\n评估训练集...")
train_final_metrics = evaluate_and_save_metrics(train_ds, 'train', model, results_dir, actual_epochs)

print("\n评估验证集...")
val_final_metrics = evaluate_and_save_metrics(val_ds, 'val', model, results_dir, actual_epochs)

print("\n评估测试集...")
test_final_metrics = evaluate_and_save_metrics(test_ds, 'test', model, results_dir, actual_epochs)

# 打印最终评估结果
print("\n最终评估结果汇总:")
print(f"训练集准确率: {train_final_metrics['accuracy']:.4f}")
print(f"训练集F1分数: {train_final_metrics['f1_score']:.4f}")
print(f"验证集准确率: {val_final_metrics['accuracy']:.4f}")
print(f"验证集F1分数: {val_final_metrics['f1_score']:.4f}")
print(f"测试集准确率: {test_final_metrics['accuracy']:.4f}")
print(f"测试集F1分数: {test_final_metrics['f1_score']:.4f}")

# 保存模型
model.save(os.path.join(results_dir, 'grape_leaf_model.h5'))

# 训练时间统计
nowtime2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
endtime = time.time()
print("\n开始时间=" + nowtime1)
print("结束时间=" + nowtime2)
print(f"运行时间：{endtime - starttime:.2f}秒")
print(f"运行时间：{(endtime - starttime) / 60:.2f}分")
print(f"运行时间：{(endtime - starttime) / 3600:.2f}时")

print(f"\n所有结果已保存至：{results_dir}")
