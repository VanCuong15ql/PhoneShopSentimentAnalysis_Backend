from transformers import AutoModel, AutoTokenizer

if __name__ == '__main__':
    # Tải mô hình và tokenizer
    model = AutoModel.from_pretrained('vinai/phobert-base')
    tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

    # Lưu mô hình và tokenizer vào thư mục địa phương
    model.save_pretrained('./phobert-base')
    tokenizer.save_pretrained('./phobert-base')

    print("Mô hình và Tokenizer đã được lưu vào thư mục './phobert-base'.")
