"""
将评估报告格式的 Text2Gloss 结果转换为标准预测格式

输入格式（当前）：
{
    "wer": ...,
    "wer_list": {
        "alignment_lst": [
            {"align_ref_lst": [...], "align_hyp_lst": [...], "align_lst": [...]},
            ...
        ]
    }
}

输出格式（标准）：
{
    "sentence_0": {"gls_hyp": "SCHON MORGEN DREISSIG GRAD"},
    "sentence_1": {...},
    ...
}
"""

import pickle
import sys

def convert_format(input_path, output_path):
    print(f"📖 Loading from: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # 检查数据结构
    if not isinstance(data, dict):
        print(f"❌ Error: Expected dict, got {type(data)}")
        sys.exit(1)
    
    # 提取 alignment_lst
    if 'wer_list' not in data:
        print("❌ Error: 'wer_list' not found in data")
        sys.exit(1)
    
    wer_list = data['wer_list']
    
    if isinstance(wer_list, dict) and 'alignment_lst' in wer_list:
        alignment_lst = wer_list['alignment_lst']
    elif isinstance(wer_list, list):
        alignment_lst = wer_list
    else:
        print(f"❌ Error: Unexpected wer_list structure: {type(wer_list)}")
        sys.exit(1)
    
    print(f"✅ Found alignment_lst with {len(alignment_lst)} samples")
    
    # 转换格式
    converted_data = {}
    skipped_count = 0
    
    for idx, item in enumerate(alignment_lst):
        sample_key = f"sentence_{idx}"
        
        if not isinstance(item, dict):
            print(f"⚠️ Warning: Sample {idx} is not a dict, skipping")
            skipped_count += 1
            continue
        
        # 优先使用 align_hyp_lst
        if 'align_hyp_lst' in item:
            raw_gloss = item['align_hyp_lst']
            
            # 过滤占位符（如 "*****"）和空字符串
            filtered_tokens = [g for g in raw_gloss if g and not all(c == '*' for c in g)]
            
            if len(filtered_tokens) == 0:
                print(f"⚠️ Warning: Sample {idx} has no valid tokens after filtering, using empty string")
                gloss_str = ""
            else:
                gloss_str = " ".join(filtered_tokens)
            
            converted_data[sample_key] = {
                "gls_hyp": gloss_str,
                "original_align_hyp_lst": raw_gloss,  # 保留原始数据用于调试
                "filtered_count": len(filtered_tokens),
                "original_count": len(raw_gloss)
            }
        else:
            print(f"⚠️ Warning: Sample {idx} has no 'align_hyp_lst', skipping")
            skipped_count += 1
    
    print(f"\n📊 Conversion Summary:")
    print(f"   Total samples: {len(alignment_lst)}")
    print(f"   Converted: {len(converted_data)}")
    print(f"   Skipped: {skipped_count}")
    
    # 统计有效 token 数量
    total_filtered = sum(item["filtered_count"] for item in converted_data.values())
    total_original = sum(item["original_count"] for item in converted_data.values())
    print(f"   Total original tokens: {total_original}")
    print(f"   Total filtered tokens: {total_filtered}")
    print(f"   Filter rate: {(1 - total_filtered/total_original)*100:.2f}% (占位符比例)")
    
    # 保存转换后的数据
    print(f"\n💾 Saving to: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(converted_data, f)
    
    # 验证输出格式
    print(f"\n🔍 Verification:")
    first_key = list(converted_data.keys())[0]
    print(f"   First sample key: {first_key}")
    print(f"   Keys: {list(converted_data[first_key].keys())}")
    print(f"   Example gls_hyp: {converted_data[first_key]['gls_hyp'][:100]}...")
    
    print(f"\n✅ Conversion completed successfully!")

if __name__ == "__main__":
    input_file = "./Data/phoenix_text2gloss_results.pkl"
    output_file = "./Data/phoenix_text2gloss_converted.pkl"
    
    convert_format(input_file, output_file)
    
    print(f"\n📝 Next steps:")
    print(f"   1. Backup original file: mv {input_file} {input_file}.backup")
    print(f"   2. Replace with converted: mv {output_file} {input_file}")
    print(f"   3. Run test: python __main__.py test ./Configs/Sign-Base.yaml --ckpt ./Models/Base-4L+4H+L1Loss/best.ckpt")
