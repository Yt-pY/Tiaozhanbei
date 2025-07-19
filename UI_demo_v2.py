import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr
from transformers import StoppingCriteriaList

# 从原始代码中导入必要的模块
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, CONV_VISION, StoppingCriteriaSub

# ===================================================================
# 步骤 1: 一次性设置和模型加载 (保持不变)
# ===================================================================
CFG_PATH = "./eval_configs/MiniGPT_3D_conv_UI_demo.yaml"
GPU_ID = 1
class Args:
    def __init__(self, cfg_path, gpu_id):
        self.cfg_path = cfg_path
        self.gpu_id = gpu_id
        self.options = None
print('正在初始化模型，请稍候...')
args = Args(CFG_PATH, GPU_ID)
cfg = Config(args)
if hasattr(cfg, 'run_cfg') and hasattr(cfg.run_cfg, 'seed'):
    print(f"在配置中找到 'run.seed'，正在设置随机种子...")
    seed = cfg.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
else:
    print("警告: 在配置文件中未找到 'run.seed'。将使用默认的随机种子。")
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0, 'pretrain_llama2': CONV_VISION_LLama2, 'pretrain': CONV_VISION}
CONV_VISION = conv_dict[model_config.model_type]
stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [torch.tensor(ids).to(f'cuda:{args.gpu_id}') for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
chat = Chat(model, device=f'cuda:{args.gpu_id}', stopping_criteria=stopping_criteria)
print(f'模型初始化完成！应用正在加载，请通过Gradio链接访问。')

# ===================================================================
# 步骤 2: Gradio 回调函数定义 (已修改)
# ===================================================================

def gradio_reset(chat_state, pc_list):
    if chat_state is not None:
        chat_state.messages = []
    if pc_list is not None:
        pc_list = []
    return (
        None,  # output (Plot)
        None,  # chatbot
        gr.update(value=None),  # point_cloud_input
        gr.update(placeholder='请先上传文件', interactive=False),  # text_input
        gr.update(value="上传并开始对话", interactive=True),  # upload_button
        None,  # chat_state
        None   # pc_list
    )

def upload_pc_v2(point_cloud_input, text_input, chat_state):
    chat_state = CONV_VISION.copy()
    pc_list = []
    
    if point_cloud_input is None:
        raise gr.Error("请先选择一个文件再点击上传！")
    
    chat.upload_pc_v2(chat_state) # 这行似乎是为特定逻辑准备的，保留
    pc_fig, pc_list = chat.encoder_pc_file(point_cloud_input, pc_list)
    
    return (
        pc_fig,  # output (Plot)
        gr.update(interactive=True, placeholder='现在可以开始提问了...'),  # text_input
        gr.update(value="对话中...", interactive=False),  # upload_button
        chat_state,
        pc_list
    )

def gradio_ask(user_message, chatbot, chat_state):
    # (此函数无需修改)
    if len(user_message) == 0:
        raise gr.Error("输入内容不能为空！")
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return '', chatbot, chat_state

def gradio_answer(chatbot, chat_state, pc_list, num_beams, temperature, max_new_tokens, max_length, min_length):
    # (此函数无需修改)
    llm_message = chat.answer(conv=chat_state, pc_list=pc_list, num_beams=num_beams, temperature=temperature, max_new_tokens=max_new_tokens, min_length=min_length, max_length=max_length)[0]
    chatbot[-1][1] = llm_message
    print(f"对话轮次-{len(chatbot)}: {chatbot[-1]}")
    return chatbot, chat_state, pc_list

# ===================================================================
# 步骤 3: UI 构建函数 (已修改)
# ===================================================================
def create_demo():
    title = """<h1 align="center">数字文物交互系统演示</h1>"""
    description_1 = """<h3>我们在MiniGPT-3D工作的基础上，在数字文物这一下游任务上进行微调，实现数字文物的人机交互</h3>"""
    
    description = """
                ##### 使用说明:
                1. 在左侧栏中点击上传区域，选择一个本地点云文件 (.ply 或 .npy)。
                2. 点击 **上传并开始对话** 按钮。
                3. 在右侧聊天框中就上传的模型开始提问。
                 """

    with gr.Blocks(title="数字文物Demo") as demo:
        gr.Markdown(title)
        gr.Markdown(description_1)
        gr.Markdown("""[[项目主页](https://tangyuan96.github.io/minigpt_3d_project_page/)]   [[论文](https://arxiv.org/pdf/2405.01413)]   [[代码](https://github.com/TangYuan96/MiniGPT-3D)]""")
        gr.Markdown(description)

        with gr.Row():
            with gr.Column(scale=1):
                point_cloud_input = gr.File(
                    file_types=[".ply", ".npy"],
                    visible=True, # 默认可见
                    label="上传点云文件 (.ply 或 .npy)"
                )
                
                with gr.Accordion("高级设置", open=False):
                    # (高级设置内容保持不变)
                    with gr.Row():
                        with gr.Column():
                            num_beams = gr.Slider(minimum=1, maximum=10, value=1, step=1, interactive=True, label="束搜索宽度")
                            temperature = gr.Slider(minimum=0.1, maximum=2.0, value=0.2, step=0.1, interactive=True, label="温度")
                        with gr.Column():
                            max_new_tokens = gr.Slider(minimum=10, maximum=200, value=60, step=10, interactive=True, label="最大回复词数")
                            max_length = gr.Slider(minimum=400, maximum=1500, value=400, step=100, interactive=True, label="最大对话长度")
                    min_length = gr.Slider(minimum=1, maximum=200, value=1, step=5, interactive=True, label="最小回复词数")
                
                with gr.Row():
                    upload_button = gr.Button(value="上传并开始对话", interactive=True, variant="primary")
                    clear = gr.Button("重启会话")

            output = gr.Plot(label="3D模型预览", scale=2)
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label='数字文物助手', height=500)
                text_input = gr.Textbox(label='用户', placeholder='请先上传文件', interactive=False)
        
        chat_state = gr.State()
        pc_list = gr.State()
        
        upload_button.click(
            upload_pc_v2,
            [point_cloud_input, text_input, chat_state],
            [output, text_input, upload_button, chat_state, pc_list]
        )
        
        text_input.submit(
            gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state]
        ).then(
            gradio_answer,
            [chatbot, chat_state, pc_list, num_beams, temperature, max_new_tokens, max_length, min_length],
            [chatbot, chat_state, pc_list]
        )

        gr.Markdown(
            """
#### Acknowledgements
[[PointLLM](https://github.com/OpenRobotLab/PointLLM/tree/master)] [[TinyGPT-V](https://github.com/DLYuanGod/TinyGPT-V)] [[MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4)]
            """
        )
        
        clear.click(
            gradio_reset, [chat_state, pc_list],
            [output, chatbot, point_cloud_input, text_input, upload_button, chat_state, pc_list],
            queue=False
        )
    return demo

# ===================================================================
# 步骤 4: 启动入口 (保持不变)
# ===================================================================
demo = create_demo()

if __name__ == "__main__":
    demo.launch()