import torch.nn as nn
import os


def load_bert(model_path):
    """
    Loads a pre-trained BERT model and sets it to evaluation mode.

    Args:
        model_path (str): Path to the pre-trained BERT model.

    Returns:
        BERT: An instance of the BERT class with the model loaded and frozen.
    """

    # 加载模型
    bert = BERT(model_path)
    bert.eval()  # Set the model to evaluation mode # 设置为评估模式
    bert.text_model.training = False  # Ensure the text model is not in training mode
    for p in bert.parameters():
        p.requires_grad = False  # Freeze all parameters
    return bert


class BERT(nn.Module):
    """
    A wrapper class for a pre-trained BERT model, including its tokenizer and text model.

    Attributes:
        tokenizer (AutoTokenizer): Tokenizer for processing input text.
        text_model (AutoModel): Pre-trained BERT model for generating embeddings.

    使用BERT模型进行文本特征提取的PyTorch实现
    """

    def __init__(self, modelpath: str):
        """
        Initializes the BERT class by loading the tokenizer and text model.

        Args:
            modelpath (str): Path to the pre-trained BERT model.
        """
        super().__init__()

        from transformers import AutoTokenizer, AutoModel
        from transformers import logging

        logging.set_verbosity_error()  # Suppress unnecessary logging
        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable parallelism warnings for tokenizers
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath)
        # Load the text model
        self.text_model = AutoModel.from_pretrained(modelpath)

    def forward(self, texts):
        """
        Processes input texts and generates embeddings using the BERT model.

        Args:
            texts (list of str): List of input text strings.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The last hidden state of the BERT model.
                - torch.Tensor: The attention mask indicating valid tokens.
        """
        # Tokenize the input texts 对文本进行tokenize
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)

        # Generate embeddings using the text model 通过BERT模型获取输出
        output = self.text_model(**encoded_inputs.to(self.text_model.device)).last_hidden_state

        # Extract the attention mask 获取注意力掩码
        mask = encoded_inputs.attention_mask.to(dtype=bool)

        return output, mask


# 测试代码
if __name__ == "__main__":
    # 加载模型
    bert_model = load_bert("bert-base-uncased")

    # 处理文本
    texts = ["Hello world!", "This is a test."]
    embeddings, attention_mask = bert_model(texts)

    # embeddings包含每个token的上下文表示
    # attention_mask标识哪些位置是有效token
