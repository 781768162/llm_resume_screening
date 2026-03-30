import os
import json
import re
import time
from dotenv import load_dotenv
from openai import OpenAI

# 加载 .env 环境变量
load_dotenv()

# 从环境变量中读取阿里云通义千问大模型参数
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3.5-plus")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))

# 初始化兼容 OpenAI SDK 格式的阿里云客户端
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def clean_json_response(raw_text: str) -> str:
    """清理大模型可能返回的 markdown 标记，提取纯 JSON"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return raw_text.strip()

def to_int_label(value):
    """将标签统一转为 0/1 整数，无法转换时返回 None"""
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if text in {"0", "1"}:
                return int(text)
    except Exception:
        pass
    return None

def main():
    cv_input_dir = r"d:\Python File\graduate_project\data\cv_data"
    jd_input_dir = r"d:\Python File\graduate_project\data\job_description"
    output_dir = r"d:\Python File\graduate_project\data\baseline_result_plus"

    print("正在读取待评估的数据集...")
    cvs = read_cvs(cv_input_dir, jd_input_dir)
    print(f"共加载了 {len(cvs)} 份简历。")

    # 临时测试：目前开启切片，只测试读取的前 2 条数据，等你测试跑通后注释掉这行即可！
    #cvs = cvs[:10]

    # 准确率统计
    total_eval = 0
    correct_eval = 0
    invalid_result = 0

    for cv_info in cvs:
        print(f"\n==============================================")
        # judge_cv 内部自带重试机制，如果彻底失败会返回 None
        res_data = judge_cv(cv_info)
        
        if not res_data:
            print(f"[{cv_info['resume_id']}] 评估彻底失败，跳过...")
            continue
        
        # 将测试集的“绝对真相(ground_truth)”保留到结果里，这是后续跑评测脚本用来对比算分的核心！
        res_data['ground_truth'] = cv_info['ground_truth']
        # 强制将结果中的匿名 ID 恢复为原本真实的 resume_id
        res_data['resume_id'] = cv_info['resume_id']

        # 统计准确率：模型输出 result vs ground_truth
        pred = to_int_label(res_data.get('result'))
        truth = to_int_label(cv_info.get('ground_truth'))
        if pred is None or truth is None:
            invalid_result += 1
            print(f"[{cv_info['resume_id']}] 警告：result 或 ground_truth 非法，跳过该条计分。")
        else:
            total_eval += 1
            if pred == truth:
                correct_eval += 1

        # 保存结果到本地
        is_saved = save_res(res_data, output_dir, cv_info['jd_folder'])
        if not is_saved:
            print(f"[{cv_info['resume_id']}] 警告：结果保存失败。")
        
        # 延时控制，防止 API 并发过高报错
        time.sleep(1)
        
    print("\n所有基线推理任务完成！")
    if total_eval > 0:
        accuracy = correct_eval / total_eval
        print("\n================ 评估统计 ================")
        print(f"有效计分样本数: {total_eval}")
        print(f"预测正确数: {correct_eval}")
        print(f"准确率 Accuracy: {accuracy:.2%}")
        print(f"非法/跳过计分样本数: {invalid_result}")
    else:
        print("\n未获得可计分样本，无法计算准确率。")


def save_res(res_data: dict, output_base_dir: str, jd_folder: str) -> bool:
    """将评估结果独立保存成 JSON 文件，按照对应岗位JD建文件夹"""
    try:
        jd_output_dir = os.path.join(output_base_dir, jd_folder)
        os.makedirs(jd_output_dir, exist_ok=True)
        
        # 保存结构：res_data 已经含有了 LLM 判定的 result 和 生成时的 ground_truth
        file_path = os.path.join(jd_output_dir, f"{res_data['resume_id']}_eval.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(res_data, f, ensure_ascii=False, indent=2)
            
        print(f"     [+] 结果已保存至: {file_path}")
        return True
    except Exception as e:
        print(f"     [-] 文件保存异常: {e}")
        return False


def read_cvs(cv_dir: str, jd_dir: str) -> list:
    """遍历简历目录，并读取简历对应的岗位JD全文本"""
    cv_list = []
    if not os.path.exists(cv_dir):
        return cv_list
        
    for jd_folder in os.listdir(cv_dir):
        folder_path = os.path.join(cv_dir, jd_folder)
        if os.path.isdir(folder_path):
            # 找到这份简历对应的原始 JD 描述文件 (文件夹名往往就是对应的 Markdown 文件名)
            jd_file_path = os.path.join(jd_dir, f"{jd_folder}.md")
            jd_text = ""
            if os.path.exists(jd_file_path):
                with open(jd_file_path, "r", encoding="utf-8") as f:
                    jd_text = f.read()
            
            # 使用前面咱们设定的规则提取如 "JD_01" 作为 ID
            target_jd_id = "_".join(jd_folder.split('_')[:2])
            
            # 遍历文件夹下所有的简历文件
            for cv_file in os.listdir(folder_path):
                if cv_file.endswith(".json"):
                    cv_path = os.path.join(folder_path, cv_file)
                    with open(cv_path, "r", encoding="utf-8") as f:
                        cv_data = json.load(f)
                        
                    #删除可能会暴露 Ground Truth 的关键字段，防止模型“作弊”
                    cv_to_llm = cv_data.copy()
                    if "ground_truth" in cv_to_llm:
                        del cv_to_llm["ground_truth"]
                    if "ground_truth_reason" in cv_to_llm:
                        del cv_to_llm["ground_truth_reason"]
                    
                    real_resume_id = cv_data.get("resume_id", cv_file.replace(".json", ""))
                    # 隐藏 ID 中的 PASS/FAIL
                    cv_to_llm["resume_id"] = "CANDIDATE_ANONYMOUS"
                    
                    cv_list.append({
                        "resume_id": real_resume_id,
                        # 传给大模型的只包含纯净的简历文本
                        "resume_text": json.dumps(cv_to_llm, ensure_ascii=False, indent=2),
                        "target_jd_id": target_jd_id,
                        "jd_text": jd_text,
                        "ground_truth": cv_data.get("ground_truth"), # 但我们自己要在后台保留这个真相
                        "jd_folder": jd_folder
                    })
    return cv_list


def judge_cv(cv_info: dict) -> dict:
    """读取 Prompt 模板，组装参数后调用模型，完成单份简历初筛推理"""
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_content = f.read()
        
    # 解析 System Prompt 与 User Prompt
    parts = prompt_content.split("---")
    system_prompt = parts[0].replace("# System Prompt", "").strip()
    user_prompt_template = parts[1].replace("# User Prompt", "").strip()

    # 将占位符替换为真实的 CV 和 JD 文本
    user_prompt = (
        user_prompt_template.replace("{{TARGET_JD_ID}}", cv_info["target_jd_id"])
        .replace("{{JD_TEXT}}", cv_info["jd_text"])
        .replace("{{RESUME_ID}}", "CANDIDATE_ANONYMOUS") # 脱敏，防止大模型从名字上看出 PASS/FAIL
        .replace("{{RESUME_TEXT}}", cv_info["resume_text"])
    )

    for attempt in range(MAX_RETRIES):
        print(f"[{cv_info['resume_id']}] 正在呼叫 {MODEL_NAME} 进行评审... (尝试 {attempt + 1}/{MAX_RETRIES})")
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, 
                response_format={"type": "json_object"},
                extra_body={"enable_thinking": False}
            )
            
            raw_output = response.choices[0].message.content
            pure_json = clean_json_response(raw_output)
            res_data = json.loads(pure_json)
            
            return res_data
            
        except json.JSONDecodeError as e:
            print(f"[{cv_info['resume_id']}] JSON 解析失败，模型未能严格按格式输出: {e}")
        except Exception as e:
            print(f"[{cv_info['resume_id']}] 阿里云 API 调用出现异常: {e}")
            time.sleep(1)
            
    return None


if __name__ == "__main__":
    main()