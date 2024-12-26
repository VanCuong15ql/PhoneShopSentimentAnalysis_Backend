import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import os
import sys
from preprocess import preprocess_fn
# Định nghĩa các khía cạnh và mức độ cảm xúc
ASPECTS = ['SCREEN', 'CAMERA', 'FEATURES', 'BATTERY', 'PERFORMANCE', 'STORAGE', 'DESIGN', 'PRICE', 'GENERAL', 'SER&ACC']
POLARITIES = {'Positive': 1, 'Negative': 2, 'Neutral': 3}
# Add imports
import torch.nn.functional as F

class ABSAModel(nn.Module):
    def __init__(self, num_aspects=10, num_classes=4):
        super().__init__()
        self.phobert = AutoModel.from_pretrained('./phobert-base')
        self.dropout = nn.Dropout(0.3)
        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=384,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Linear(768, num_aspects * num_classes)
        self.num_aspects = num_aspects
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state

        x = self.dropout(x)
        lstm_out, _ = self.bilstm(x)

        mask_expanded = attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
        sum_embeddings = torch.sum(lstm_out * mask_expanded, 1)
        avg_embeddings = sum_embeddings / attention_mask.sum(1).unsqueeze(-1).float()

        logits = self.classifier(avg_embeddings)
        logits = logits.view(-1, self.num_aspects, self.num_classes)

        if labels is not None:
            # Convert labels to Long type for cross entropy
            labels = labels.long()
            loss = F.cross_entropy(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        return logits


def predict_sentiment(text, model, tokenizer):
    # Tiền xử lý văn bản
    processed_text = preprocess_fn(text)

    # Mã hóa văn bản
    encoding = tokenizer(
        processed_text,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Chuyển dữ liệu sang GPU (nếu có)
    input_ids = encoding['input_ids'].to('cpu')
    attention_mask = encoding['attention_mask'].to('cpu')

    # Dự đoán
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        predicted_labels = torch.argmax(logits, dim=2)

    # Lấy kết quả
    predicted_aspects = []
    for i, aspect_label in enumerate(predicted_labels[0]):
        if aspect_label != 0:  # 0 là nhãn 'None'
            aspect = ASPECTS[i]
            polarity = list(POLARITIES.keys())[list(POLARITIES.values()).index(aspect_label.item())]
            predicted_aspects.append((aspect, polarity))

    return predicted_aspects


def main():
    # Kiểm tra nếu có đối số dòng lệnh
    if len(sys.argv) < 2:
        sys.exit(1)

    # Lấy văn bản từ đối số dòng lệnh
    text = sys.argv[1]

    # Tải model và tokenizer
    model_path = os.path.join(os.getcwd(), 'model', 'best_model.pt')
    # Sử dụng map_location để tải model lên CPU
    model = ABSAModel().to('cpu')  # Khởi tạo mô hình
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    # Tải tokenizer
    PHOBERT_NAME = "./phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_NAME)

    # Dự đoán cho văn bản đầu vào
    predicted_aspects = predict_sentiment(text, model, tokenizer)

    # Chuyển đổi kết quả thành định dạng 'LABEL:SENTIMENT'
    formatted_output = [f"{label}:{sentiment}" for label, sentiment in predicted_aspects]

    #print(f"Văn bản: {text}")
    for output in formatted_output:
        print(output)


if __name__ == '__main__':
    main()
