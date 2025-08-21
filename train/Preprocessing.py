import json
def text_to_jsonl_dataset(metin, chunk_size=1024):
    chunks = [metin[i:i+chunk_size] for i in range(0, len(metin), chunk_size)]
    jsonl_list = []
    for chunk in chunks:
        item = {"prompt": chunk}
        json_line = json.dumps(item, ensure_ascii=False)
        jsonl_list.append(json_line)    
    print(f"[+] Dataset {len(chunks)} satır olarak RAM'de üretildi.")
    return jsonl_list

def prepare_dataset(dataset,tokenizer, max_length=1024):
    def tokenize_function(examples):
        tokens = tokenizer(
            examples["prompt"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens
    return dataset.map(tokenize_function, batched=True)

def pdf_to_text(pdf):
    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    return text