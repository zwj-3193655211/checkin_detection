# 晨读晨练签到检测系统 - 架构图

## 1. 整体系统架构图

```mermaid
flowchart LR
    subgraph 输入层
        A[图片输入<br/>JPG/PNG] --> B[CLIP预处理<br/>224x224]
    end
    
    subgraph 特征提取层
        B --> C[CLIP ViT-B/32<br/>视觉Transformer]
        C --> D[图像块嵌入<br/>Patch Embedding]
        D --> E[多头注意力<br/>12头]
        E --> F[前馈网络<br/>FFN]
        F --> G[512维特征向量]
    end
    
    subgraph 隐藏层
        G --> H[MLP主分类器<br/>512→256→128→2]
        G --> I[MLP特征预测器<br/>512→512→256→128→11]
    end
    
    subgraph 输出层
        H --> J[Softmax<br/>晨读/晨跑/异常]
        I --> K[Sigmoid<br/>11维特征]
    end
    
    J --> L[三支决策]
    K --> L
    L --> M[结果输出<br/>自动通过/待审核]
```

---

## 2. MLP主分类器详细结构

```mermaid
flowchart LR
    subgraph 输入层
        A[512维特征向量<br/>CLIP输出]
    end
    
    subgraph 隐藏层1
        A --> B[Linear 512→256]
        B --> C[ReLU激活]
        C --> D[Dropout 0.3]
    end
    
    subgraph 隐藏层2
        D --> E[Linear 256→128]
        E --> F[ReLU激活]
        F --> G[Dropout 0.3]
    end
    
    subgraph 输出层
        G --> H[Linear 128→2]
        H --> I[Softmax]
        I --> J[分类概率<br/>晨读/晨跑]
    end
```

---

## 3. MLP特征预测器详细结构

```mermaid
flowchart LR
    subgraph 输入层
        A[512维特征向量<br/>CLIP输出]
    end
    
    subgraph 隐藏层1
        A --> B[Linear 512→512]
        B --> C[LayerNorm]
        C --> D[SiLU激活]
        D --> E[Dropout 0.25]
    end
    
    subgraph 隐藏层2
        E --> F[Linear 512→256]
        F --> G[LayerNorm]
        G --> H[SiLU激活]
        H --> I[Dropout 0.25]
    end
    
    subgraph 残差融合
        E --> J[残差投影 512→128]
        I --> K[Linear 256→128]
        K --> L[残差相加]
    end
    
    subgraph 输出层
        L --> M[Linear 128→11]
        M --> N[Sigmoid]
        N --> O[11维特征概率<br/>人脸/蓝色桌子/教室/...]
    end
```

---

## 4. CLIP ViT-B/32 内部结构

```mermaid
flowchart LR
    subgraph 输入
        A[224×224×3<br/>RGB图片]
    end
    
    subgraph Patch嵌入层
        A --> B[分成16×16块<br/>49个patch]
        B --> C[线性投影<br/>Patch Embedding]
    end
    
    subgraph Transformer编码器
        C --> D[+位置编码]
        D --> E[多头自注意力<br/>12个头]
        E --> F[残差连接+层归一化]
        F --> G[前馈网络<br/>2048隐藏维]
        G --> H[残差连接+层归一化]
    end
    
    subgraph 输出
        H --> I[CLS_token]
        I --> J[512维特征向量]
    end
```

---

## 5. 三支决策流程图

```mermaid
flowchart TD
    A[图片输入] --> B[CLIP特征提取]
    B --> C[MLP主分类器]
    C --> D{置信度判断}
    D -->|≥0.80| E[MLP特征预测器]
    D -->|<0.80| F[待审核]
    E --> G{特征匹配数}
    G -->|≥3| H[自动通过]
    G -->|<3| I[待审核]
    H --> J[输出结果]
    F --> J
    I --> J
```

---

## 6. 训练流程图

```mermaid
flowchart TD
    A[加载CLIP特征<br/>CSV文件] --> B[加载标注数据<br/>labels.json]
    B --> C[数据划分<br/>70%训练/10%验证/20%测试]
    C --> D{训练轮次}
    D -->|Epoch 1-N| E[前向传播]
    E --> F[计算损失]
    F --> G[反向传播<br/>梯度下降]
    G --> H[参数更新<br/>Adam优化器]
    H --> D
    D -->|训练完成| I[保存模型]
    I --> J[mlp_classifier.pt<br/>mlp_features.pt]
```

---

## 7. 信息流完整路径图

```mermaid
flowchart LR
    subgraph "图片输入"
        A[原始图片<br/>JPG/PNG]
    end
    
    subgraph "特征提取(CLIP)"
        A --> B[图像预处理<br/>Resize+CenterCrop]
        B --> C[ViT-B/32<br/>Transformer]
        C --> D[512维向量]
    end
    
    subgraph "MLP主分类器(隐藏层)"
        D --> E[Linear 512→256]
        E --> F[ReLU+Dropout]
        F --> G[Linear 256→128]
        G --> H[ReLU+Dropout]
        H --> I[Linear 128→2]
    end
    
    subgraph "输出层"
        I --> J[Softmax]
        J --> K[概率分布<br/>晨读/晨跑]
    end
    
    subgraph "三支决策"
        K --> L{置信度判断}
        L -->|高|M[自动通过]
        L -->|低|N[待审核]
    end
```

---

## 8. 项目文件结构图

```mermaid
mindmap
  root((晨读晨练检测系统))
    data
      clip_features_cpu.csv
      labels.json
      mlp_classifier.pt
      mlp_features.pt
      split_config.json
    src
      checkin_system.py
      models/mlp.py
    scripts
      feature_label_tool.py
    train
      train_mlp_binary.py
    outputs
      reports/*.json
    docs
      README.md
      TECHNICAL_DOCUMENTATION.md
```

---

## 9. 决策规则流程图

```mermaid
flowchart TD
    start([开始]) --> A[输入图片]
    A --> B[CLIP特征提取]
    B --> C[MLP主分类器预测]
    C --> D{置信度<0.80?}
    D -->|是| E[待审核]
    D -->|否| F[MLP特征预测]
    F --> G[匹配特征数<3?]
    G -->|是| E
    G -->|否| H[晨跑特征≥5?]
    H -->|是| I[自动通过]
    H -->|否| J[晨读特征≥3?]
    J -->|是| I
    J -->|否| E
    I --> end1([结束])
    E --> end2([结束])
```

---

## 10. 完整训练数据流

```mermaid
sequenceDiagram
    participant 数据集 as 数据集
    participant CLIP as CLIP特征提取
    participant Train as 训练过程
    participant Model as 模型保存
    participant Test as 测试推理
    
    数据集->>CLIP: 1.提取特征
    CLIP->>Train: 2.512维向量
    Train->>Train: 3.前向传播
    Train->>Train: 4.计算损失
    Train->>Train: 5.反向传播
    Train->>Train: 6.梯度下降
    Train->>Model: 7.保存模型
    Model->>Test: 8.加载模型
    Test->>Test: 9.推理预测
    Test->>用户: 10.输出结果
```
