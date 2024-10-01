import torch
from PIL import Image
from torchvision import transforms
import os
import timm
import torch.nn as nn



# 從 timm 加載 model
print("start")
model = timm.create_model('convnextv2_base', pretrained=False)

if torch.cuda.is_available():
    model.load_state_dict(torch.load(r"F:\Lab\share\GUI\GUI_0417\CNN_model\convnext\original\trained_model_weights.pth"))
else:
    model.load_state_dict(torch.load(r"F:\Lab\share\GUI\GUI_0417\CNN_model\convnext\original\trained_model_weights.pth", map_location=torch.device('cpu')))



# 修改全連接層的輸出形狀
num_ftrs = model.head.fc.in_features
model.head.fc = nn.Linear(num_ftrs, 3)  # 將2替換為你的類別數量

# 從 trained_model_weights.pth 加載自己訓練的權重
print("start")
model.load_state_dict(torch.load(r"F:\Lab\share\GUI\GUI_0417\CNN_model\convnext\original\trained_model_weights.pth"))

# 創建一個字典來對應數字標籤到類別名稱
class_label_mapping = {
    0:"apical lesion",
    1:"normal",
    2:"peri-endo"
}


# 指定測試資料夾路徑
testFolderPath = r"F:\Lab\share\GUI\GUI_0417\origin";

# 取得資料夾中所有影像的檔案列表
imgFiles = [f for f in os.listdir(testFolderPath) if f.endswith('.jpg')]

# 循環處理每個影像
for i in range(len(imgFiles)):
    # 讀取影像
    imgPath = os.path.join(testFolderPath, imgFiles[i])
    img = Image.open(imgPath)

    # 調整影像大小
    img_resized = img.resize((240, 240))

    # 將影像轉換為 PyTorch 張量
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 可以添加其他预处理步骤，例如归一化
    ])
    img_tensor = transform(img_resized).unsqueeze(0)  # 添加批次维度

    # 使用模型進行分類和獲取預測標籤
    with torch.no_grad():
        output = model(img_tensor)
        # 將模型的輸出通過 softmax 函數計算機率
        probabilities = torch.softmax(output, dim=1)
        # 獲取預測機率最高的類別索引和機率值
        max_prob, predicted_class = torch.max(probabilities, 1)
        label_index = predicted_class.item()
        predicted_prob = max_prob.item()

    label = class_label_mapping.get(label_index, 'unknown')  # 轉換為字符串

    # 顯示分類結果和預測機率
    print(f'Image: {imgFiles[i]}, Predicted Label: {label}, Predicted Probability: {predicted_prob}')


    # 在這裡，你可以選擇將結果儲存到一個陣列或檔案中，以便進一步分析


print("-"*60)

