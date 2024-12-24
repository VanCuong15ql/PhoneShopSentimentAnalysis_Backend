import os
import sys
import torch
from torch import nn
from transformers import XLMRobertaTokenizer, XLMRobertaModel
from TorchCRF import CRF

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()

        # Lớp nhúng từ XLM-RoBERTa
        self.xlm_roberta = XLMRobertaModel.from_pretrained('xlm-roberta-base')

        # Đóng băng các tham số của mô hình tiền huấn luyện
        for param in self.xlm_roberta.parameters():
            param.requires_grad = False

        # Lớp BiLSTM
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        # Lớp tuyến tính để ánh xạ đầu ra BiLSTM
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        # CRF Layer
        self.crf = CRF(num_classes, batch_first=True)

        # Store number of classes as an attribute
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask, labels=None):
        # Lấy embedding từ XLM-RoBERTa
        roberta_output = self.xlm_roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Lấy hidden states cuối cùng
        lstm_input = roberta_output.last_hidden_state

        # Chạy qua BiLSTM
        lstm_out, _ = self.bilstm(lstm_input)

        # Ánh xạ qua lớp tuyến tính
        emissions = self.fc(lstm_out)

        # Tạo mask để loại trừ padding
        mask = attention_mask.type(torch.bool)

        # Nếu có nhãn, tính loss sử dụng CRF
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
            return loss

        # Nếu không có nhãn, trả về các nhãn dự đoán
        return self.crf.decode(emissions, mask=mask)

class PhoneAspectPredictor:
    def __init__(self, checkpoint_path, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

        # Load the trained model
        self.model = BiLSTM_CRF(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=768,
            hidden_dim=256,
            num_classes=31
        )

        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)

        # Extract model state dict from checkpoint
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def _load_label_map(self):
        # Load label map from a file or define it here
        return {'BATTERY#Negative': 1, 'BATTERY#Neutral': 2, 'BATTERY#Positive': 3, 'CAMERA#Negative': 4, 'CAMERA#Neutral': 5, 'CAMERA#Positive': 6, 'DESIGN#Negative': 7, 'DESIGN#Neutral': 8, 'DESIGN#Positive': 9, 'FEATURES#Negative': 10, 'FEATURES#Neutral': 11, 'FEATURES#Positive': 12, 'GENERAL#Negative': 13, 'GENERAL#Neutral': 14, 'GENERAL#Positive': 15, 'OTHERS': 16, 'PERFORMANCE#Negative': 17, 'PERFORMANCE#Neutral': 18, 'PERFORMANCE#Positive': 19, 'PRICE#Negative': 20, 'PRICE#Neutral': 21, 'PRICE#Positive': 22, 'SCREEN#Negative': 23, 'SCREEN#Neutral': 24, 'SCREEN#Positive': 25, 'SER&ACC#Negative': 26, 'SER&ACC#Neutral': 27, 'SER&ACC#Positive': 28, 'STORAGE#Negative': 29, 'STORAGE#Neutral': 30, 'STORAGE#Positive': 31}

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        with torch.no_grad():
            predictions = self.model(input_ids, attention_mask)

        # Convert predictions to labels
        id_to_label = {v: k for k, v in self._load_label_map().items()}
        predicted_labels = []

        for pred_id in predictions[0]:
            if pred_id in id_to_label and id_to_label[pred_id] != 'OTHERS':
                predicted_labels.append(id_to_label[pred_id])

        return list(set(predicted_labels))

def main():
    # Specify the path to your checkpoint file
    checkpoint_path = os.path.join(os.getcwd(), 'model', 'checkpoint_epoch_10.pth')

    try:
        # Initialize predictor
        predictor = PhoneAspectPredictor(checkpoint_path)

        # Check if text is passed as an argument
        if len(sys.argv) > 1:
            text = sys.argv[1]
            aspects = predictor.predict(text)
            for aspect in aspects:
                print(aspect)
        else:
            # Test with some example texts
            test_texts = [
                "Điện thoải ổn. Facelock cực nhanh, vân tay ôk , màn hình lớn, pin trâu ( liên quân , Zalo, YouTube ) một ngày mất khoảng 45 % ) tuy chỉ chip 439 nhưng rất mượt. Đa nhiệm khá ổn",
                "Mình mới mua vivo91c. Tải ứng dụng ,games nhanh. Có cái không hài lòng là cài hình nền khóa màn hình không được. Hay tại mình chưa biết hết chức năng của nó. Tư vấn viên nhiệt tình",
                "Xấu đẹp gì ko biết nhưng rất ưng TGdđ phục vụ rất tuyệt vời, mua ở TGdđ mòn đít và sẽ mua dài dài ủng hộ TGDđ",
            ]

            for text in test_texts:
                aspects = predictor.predict(text)
                print(f"\nText: {text}")
                print(f"Predicted aspects: {aspects}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()