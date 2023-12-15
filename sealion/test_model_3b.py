# please use transformers 4.34.1
from ast import main
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = 'aisingapore/sealion3b'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, trust_remote_code=True)

if __name__ == '__main__':
    context = []
    while True:
        command = 'Hello'
        tokens = tokenizer(command, return_tensors='pt')
        print(tokens)
        output = model.generate(tokens['input_ids'], max_new_tokens=200)
        print('Response: ' +
              tokenizer.decode(output[0], skip_special_tokens=True))
        print('\n\n')
        break
