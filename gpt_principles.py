from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import random

modelname = "EleutherAI/gpt-neo-2.7B"
device = "cuda:1"

tokenizer = AutoTokenizer.from_pretrained(modelname)
model = AutoModelForCausalLM.from_pretrained(modelname)
model.to(device)


def inference(prompt, temperature, max_length):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=temperature,
        max_length=max_length,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text


def generate(prompt, temp=0.8, max_len=150):
    generated_text = inference(prompt, temperature=temp, max_length=max_len)
    generated_text = generated_text[len(prompt):]
    principles = generated_text.split("###")

    def keep_heuristics(principle):
        return (":" in principle and
                len(principle.split(":")[0]) > 10 and
                len(principle.split(":")[0].split()) > 1
        )
    
    p = [p.strip() for p in principles if keep_heuristics(p)][:-1]
    #for p_ in p: print(p_)
    return p


prime = '''
Occam's razor: "entities should not be multiplied beyond necessity"
###
Hofstadter's Law: "It always takes longer than you expect, even when you take into account Hofstadter's Law."
###
Pareto Principle: "roughly 80% of consequences come from 20% of causes"
###
'''.strip()


with open("gpt_principles.txt", "w") as df:
    for i in tqdm(range(1000)):
        principles = generate(prime, temp=random.uniform(0.8, 1.0))
        for p in principles:
            df.write(p + "\n")
