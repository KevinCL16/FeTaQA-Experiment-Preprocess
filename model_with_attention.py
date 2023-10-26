import torch
from transformers import T5ForConditionalGeneration, T5Config
from transformers import T5Tokenizer


class T5TableQA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq2seqGeneration = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.linear = torch.nn.Linear(768, 512)
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

    def forward(self, input_representation, labels):
        input_representation = self.linear(input_representation)
        output = self.seq2seqGeneration(inputs_embeds=input_representation, labels=labels)
        return output.loss

    def generate(self, input_ids):
        outputs = self.seq2seqGeneration.generate(input_ids)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)