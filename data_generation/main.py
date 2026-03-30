import os
import time
from pipeline import generate_resume

def main():
    # 设定目录
    jd_dir = r"d:\Python File\graduate_project\data\job_description"
    output_base_dir = r"d:\Python File\graduate_project\data\cv_data"
    
    # 获取所有的JD md文件
    if not os.path.exists(jd_dir):
        print(f"找不到JD目录: {jd_dir}")
        return
        
    jd_files = [f for f in os.listdir(jd_dir) if f.endswith(".md")]
    jd_files.sort()
    
    # 临时测试：切片限制只测试第一个JD，节约 token
    #jd_files = jd_files[:1]
    
    NUM_PASS = 20  # 每个岗位生成5份通过的简历
    NUM_FAIL = 20  # 每个岗位生成5份淘汰的简历
    
    total_generated = 0
    start_time = time.time()
    
    for filename in jd_files:
        # 提取 "JD_01_蚂蚁集团_Agent工程师" 作为文件夹名
        jd_name_no_ext = filename.replace(".md", "")
        # 我们提取 "JD_01" 作为 target_jd_id
        target_jd_id = "_".join(filename.split('_')[:2])
        
        print(f"\n==============================================")
        print(f"开始处理岗位: {filename} (ID: {target_jd_id})")
        print(f"==============================================")
        
        # 为当前 JD 准备目标存储文件夹路径
        jd_output_dir = os.path.join(output_base_dir, jd_name_no_ext)
        
        filepath = os.path.join(jd_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            jd_text = f.read()
            
        # 1. 生成通过的简历
        for i in range(1, NUM_PASS + 1):
            resume_id = f"RES_{target_jd_id}_PASS_{i:03d}"
            print(f"---> 正在生成样本: {resume_id} (标签: 1) ...")
            result = generate_resume(
                jd_text=jd_text,
                target_jd_id=target_jd_id,
                resume_id=resume_id,
                match_label=1,
                output_dir=jd_output_dir
            )
            if result:
                total_generated += 1
            time.sleep(1)

        # 2. 生成淘汰的简历
        for i in range(1, NUM_FAIL + 1):
            resume_id = f"RES_{target_jd_id}_FAIL_{i:03d}"
            print(f"---> 正在生成样本: {resume_id} (标签: 0) ...")
            result = generate_resume(
                jd_text=jd_text,
                target_jd_id=target_jd_id,
                resume_id=resume_id,
                match_label=0,
                output_dir=jd_output_dir
            )
            if result:
                total_generated += 1
            time.sleep(1)
            
    end_time = time.time()
    print(f"\n==============================================")
    print(f"批量生成完毕！共成功生成 {total_generated} 份模拟简历。")
    print(f"总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()