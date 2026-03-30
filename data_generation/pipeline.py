import os
import json
import re
import time
import random
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

# --- 多样性控制要素池 ---
PREV_INDUSTRY = ["一线互联网大厂/顶尖金融机构", "传统银行/老牌金融机构", "新兴互联网金融公司", "国内独角兽公司", "顶尖外企"]

PASS_LOGIC = [
    # 严重符合 (Strong Pass)
    "完美对口：技术栈与业务经验100%匹配，有核心项目主导经历。",
    "降维打击：背景资历突出，技术与能力均远超JD期望标准。",
    # 刚好擦边 (Marginal Pass)
    "业务偏科：底层技术极其扎实，但金融垂直业务经验稍显单薄，但能够胜任。",
    "跨界契合：来自跨行领域，但近期主导过与JD核心痛点高度一致的复杂项目。",
    "技术专精：缺乏部分JD要求的边缘非核心工具经验，但在最核心的架构支撑和算法上有极深的造诣。"
]

FAIL_LOGIC = [
    # 严重不符 (Strong Fail)
    "方向脱节：主攻技术方向与JD核心需求毫无关联（如JD要底层网络，简历却全是前端业务）。",
    "技术陈旧：有对口行业经验，但使用的技术栈严重落后于现代架构要求。",
    "空心造假：大量堆砌高级架构词汇与大厂背景，但毫无项目落地数据和细节，缺乏真实沉淀。",
    # 刚好擦边 / 强混淆 (Marginal Fail)
    "缺乏业务：底层技术非常完备，但完全体现不出JD要求的金融或垂直业务场景的痛点解决经验。",
    "大厂边缘：深度参与过对口大厂核心项目，但暴露其只是边缘辅助、纯打杂或外围运维，无核心贡献。",
    "套话包装：通篇充斥“赋能、拉齐、抓手”等管理套话，严重掩饰其硬核代码能力和底层技术的匮乏。"
]

def clean_json_response(raw_text: str) -> str:
    """清理大模型可能返回的 markdown 标记，提取纯 JSON"""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    return raw_text.strip()

def generate_resume(jd_text: str, target_jd_id: str, resume_id: str, match_label: int, output_dir: str = None) -> dict:
    """
    根据 JD 和期望的标记(0或1)，结合动态组合的控制变量，调用通义千问并返回/保存生成的 JSON 简历。
    """
    # 动态载入 prompt 模板
    prompt_path = os.path.join(os.path.dirname(__file__), "prompt.md")
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_content = f.read()

    # 载入 JSON Schema 模板
    schema_path = os.path.join(os.path.dirname(__file__), "schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        resume_schema = json.load(f)

    # 将 Prompt 分隔成 System 和 User 两部分
    parts = prompt_content.split("---")
    system_prompt = parts[0].replace("# System Prompt", "").strip()
    user_prompt_template = parts[1].replace("# User Prompt", "").strip()

    # --- 动态组合多样性控制方向 ---
    industry = random.choice(PREV_INDUSTRY)
    
    if match_label == 1:
        match_label_text = "1 (通过)"
        logic = random.choice(PASS_LOGIC)
    else:
        match_label_text = "0 (淘汰)"
        logic = random.choice(FAIL_LOGIC)
        
    diversity_prompt = (
        f"【硬性门槛强锁定指令】（最高优先级！！！）：\n"
        f"无论接下来要求这名候选人是通过还是淘汰，你**必须保证该候选人的「工作年限（工龄）」和「教育学历」严格等于或略高于 JD 的最低要求！** 绝不允许生成低于 JD 年限和学历门槛的简历。我们要测试的是 HR 模型对「业务深度」的理解，绝不能让它因为极其明显的工龄不足或学历不够而投机取巧地直接判负。\n\n"
        f"【基础画像参考】：\n"
        f"- 历史背景倾向：{industry}\n"
        f"【核心约束逻辑（决定其含金量的唯一标准）】：{logic}\n\n"
        f"【生成要求】：\n"
        f"请基于上述指令，编造一份逻辑极度自洽的职业履历。通过或淘汰的原因只能体现在“项目深浅”、“业务技术匹配度”以及“核心职责的话语权”上。不得依靠硬性指标去淘汰！"
    )
        
    print(f"[{resume_id}] 多样性策略: [{industry}] --> [逻辑] {logic}")

    # 替换变量
    user_prompt = (
        user_prompt_template.replace("{{TARGET_JD_ID}}", target_jd_id)
        .replace("{{JD_TEXT}}", jd_text)
        .replace("{{RESUME_ID}}", resume_id)
        .replace("{{MATCH_LABEL}}", match_label_text)
        .replace("{{DIVERSITY_PROMPT}}", diversity_prompt)
    )

    for attempt in range(MAX_RETRIES):
        print(f"[{resume_id}] 正在调用 {MODEL_NAME} (目标标签: {match_label})... (尝试 {attempt + 1}/{MAX_RETRIES})")
        try:
            # 阿里云提供兼容模式，可直接用 client.chat.completions 开发
            # 我们直接使用最稳妥的 JSON Mode (即 "type": "json_object")，并通过 Prompt 提供强约束
            # 强制关闭思考模式，避免大模型“自问自答”导致速度慢及额外幻觉介入
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
                extra_body={
                    "enable_thinking": False
                }
            )
            
            # 使用 Structured Output 后，通常输出的就是干净的 JSON 字符串
            raw_output = response.choices[0].message.content
            pure_json = clean_json_response(raw_output)
            
            # 验证返回内容是否符合 JSON
            resume_data = json.loads(pure_json)
            print(f"[{resume_id}] 生成成功！")
            
            # 如果指定了输出目录，则自动负责写入文件
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                target_file = os.path.join(output_dir, f"{resume_id}.json")
                with open(target_file, 'w', encoding='utf-8') as outfile:
                    json.dump(resume_data, outfile, ensure_ascii=False, indent=2)
                print(f"[{resume_id}] 文件已保存至: {target_file}")
                
            return resume_data
            
        except json.JSONDecodeError as e:
            print(f"[{resume_id}] 解析 JSON 失败，模型输出格式不规范: {e}")
        except Exception as e:
            print(f"[{resume_id}] API 调用出错: {e}")
            
    print(f"[{resume_id}] 达到最大重试次数，生成失败。")
    return None

if __name__ == "__main__":
    # 模拟一份输入 JD 文本做单点测试
    sample_jd = """
    微众银行-网络技术工程师
    工作职责：
    1、根据实际业务需求及业界趋势对生产网络架构规划，并进行优化工作；
    2、对数据中心、城域网、骨干网络定期进行优化分析，并且制定网络优化方案；
    3、对网络运营中已经发生的故障进行协助分析，积极寻找下一阶段网络可以优化和改善的解决方案，并且制定网络优化方案；
    4、推进网络AI ops及AI infra设计并落地，推动周边系统和相关团队完成自动化工具开发，持续提升海量金融服务体验；
    5、对网络整个生命周期进行精细化管理，不断优化配套网络规划、建设、运营流程。
    工作要求：
    1、3年以上网络架构工作经验，互联网、金融或运营商相关背景；
    2、精通STP、TCP/IP、OSPF、BGP、ISIS、MPLS VPN／TE、VXLAN等协议；
    3、熟悉Linux系统管理、Web系统架构、HTTP协议、容器网络原理；
    4、熟练使用Python\Shell等编程语言进行自动化脚本处理；
    5、熟悉大规模IDC及金融生产网络架构、国内主流运营商网络结构、光通信、网络安全等；
    6、熟悉Cisco、Hillstone、H3C、华为等主流网络厂商产品技术及产品架构；
    7、具备大规模的数据中心网络、骨干网络建设和运营项目经验优先；
    8、具有良好的沟通能力，良好的书面和口头表达能力，较高的文档撰写水平；
    9、思维缜密，逻辑性强，具有挑战精神和前瞻性，良好的团队协作精神；
    10、具备良好的抗压能力，积极主动完成跟进项目，做好个人计划管理。
    """
    
    start_time = time.time()
    out_dir = r"d:\Python File\graduate_project\data\cv_data"

    # 测试生成一个简历
    result = generate_resume(
        jd_text=sample_jd, 
        target_jd_id="JD_TEST_001", 
        resume_id="RES_TEST_001", 
        match_label=0,
        output_dir=out_dir
    )
    
    end_time = time.time()
    
    if result:
        print("\n最终生成的简历结构：")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        print(f"\n[耗时分析] 接口调用与数据解析总计耗时: {end_time - start_time:.2f} 秒")