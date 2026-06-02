from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json
from globalenv import *
import pdb  

anth_all_path = "./Dataset/{}.jsonl"

class UniDataset(Dataset):
    def __init__(self,
                 task:str, 
                 batch_size:int=50,
                 model_path:str=MODEL,
                 inst_template:str=INST_TEMPLATE,
                 ans_prefix:str=INST_SYS_ANS_PREFIX_alt1,): 

        self.task = task
        self.batch_size = batch_size
        self.inst_template = inst_template
        self.ans_prefix = ans_prefix


        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.keys = []
        self.prompt_lens = []
        self.data = self.__load_data()


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __tok(self, input):
        return self.tokenizer(input,padding="longest",return_tensors="pt")
    
    def __load_data(self):
        return self.__load_anth_data()
    
    
    def __load_anth_data(self):
        data_toks = []
        data_path = anth_all_path.format(self.task)

        with open(data_path, 'r') as f:
            data = [json.loads(line.strip()) for line in f]
        
        data_len = len(data)

        for i in tqdm(range(data_len), desc="process data"):
            cur_data = []
            
            prompt = data[i]["harmful_input"]
            safe_res = data[i]["safe_response"]
            unsafe_res = data[i]["unsafe_response"]
            responses = [safe_res, unsafe_res]
            self.keys.append(1) 

            for res in responses:
                formatted_text = self.inst_template.format(
                    prompt, res)
                cur_data.append(formatted_text)
            
            cur_data_toks = self.__tok(cur_data)
            data_toks.append(cur_data_toks)
        return data_toks

