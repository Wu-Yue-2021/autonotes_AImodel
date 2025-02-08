# This file is used to call api of GLM model.
# 使用了uv来管理这个文件的运行环境
from pprint import pprint


import time
start_time = time.time()
time_now=time.strftime("%Y%m%d%H%M", time.localtime())

# 设置输入输出文件
file_name_nopostfix = "超长讲座part3"
input_file = f'files/{file_name_nopostfix}.txt'
# 输出放代码结尾了


# 设置模型及网址
model1_name='glm-4-air-0111'#model2_name在后面
base_url="https://open.bigmodel.cn/api/paas/v4/"



# 读取文件
# with open(input_file, 'r', encoding='utf-8') as f:
#     record_text = f.read()
with open(input_file, 'r', encoding='utf-8') as f:#判断切换模型
    n=len(f.read())
    print("输入tokens数：",n)
    if n>60000:#正常语速 约3.2-3.6万/小时（超绝宋浩2万） 粗测下来1.5小时后会超过air的记忆区，但也有50分钟超的情况
        model_name='glm-4-long'
with open(input_file, 'r', encoding='utf-8') as f:#输入改成列表
    record_text_l = f.readlines()
# print(record_text_l)
'''# 转换成元组列表（非必要）
record_text = []
for _ in record_text_l:
    # _ = _.replace('\n','')
    record_text.append((_.split('\t')[0],_.split('\t')[1]))
    # record_text.append((_.split(']')[0].replace('[',''),_.split(']')[1]))
'''
record_text = record_text_l
# print(record_text)
time0 = time.time()
print(f"读取输入文件“{file_name_nopostfix}”成功，当前步骤耗时：", f"{time0-start_time:.2f}s",f"总耗时：{time0-start_time:.2f}s")



# 大模型部分
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage


model_str = ChatOpenAI(
    model=model1_name,
    openai_api_base=base_url,
    max_tokens=4095,
)
prompt_template_str = ChatPromptTemplate.from_messages(
    [   #SystemMessage(content="hello"),
        ('system','用户以列表形式输入一堂课的录音文本，列表中每一项前半部分为这句话的时间，后半部分为这句话的文本，你被要求先优化识别错误的内容，然后根据处理后的文本，以文本形式输出一个课堂笔记，内容是：1.课程主题。2.课程摘要。3.详细的知识点笔记(每个知识点包含名称、重要程度（整数1-5，5代表最重要）、然后由子要点阐述解释或例子（子要点如定义、应用、证明等），每个子要点的标题前要有这个概念在录音文本中第一次被提到的时间（即这个子要点的开始时间,要仔细审查），每个要点后附这个子要点对应的全部文本，即不要有任何改动的原文本(注意例题也算一个子要点)））。4.几个关键词及对应的参考网站。（可以分开写但要确保覆盖全部时间全部要点）'
                  '重要程度规则如下：5.科目重点：这门课的核心内容，教师强调为这门科目的最重要内容之一，用了超过20分钟详细讲解该知识点的推导过程、应用场景及例题剖析，需要掌握复杂灵活应用的方法的知识点，如在高等数学中的微积分基本定理、线性代数中矩阵的秩，应用贯穿整个科目。课程重点最多只有一个；4.单元重点：难度及重要性不如课程重点，但与本单元核心内容紧密相关，教师多次强调重要/考试重点等的知识点，单知识点讲解时长超过10分钟，同时会涉及一定难度的灵活应用，要求学生熟练掌握并能较好运用的知识点，如线性代数中的矩阵乘法部分，是理解后续矩阵特征值、特征向量等知识的基础；3.基础重点：是后续学习其他重要知识的基础铺垫，教师讲解时长不到十分钟，通常会结合简单例题进行说明，要求学生理解并能够进行基础应用，为深入学习后续内容必备的知识点，如高等数学的极限内容，是后续学习导数、积分等知识的必备基础；2. 次要知识点：对整体知识架构有一定补充作用，教师讲解时间相对较短，可能仅提及概念或简单提及应用场景，一般不会深入展开应用，要求学生了解即可的知识点,如高等数学中方向导数的几何意义；1.了解知识点：在本节课中提及较少，与核心内容关联性较弱，仅作为拓展信息或背景知识，无需花费过多精力去深入探究的知识点，如某个领域的发展历程；重要程度为1也可能是难度过大当前无需掌握的知识点，如高等数学中含参变量的积分。'
                  '示例如下：主题：二阶行列式和三阶行列式的定义及其计算；摘要：本节课首先通过一个二元一次方程组的例子，引出了二阶行列式的定义和计算方法。接着，介绍了三阶行列式的定义和计算方法，包括对角线法则。最后，讨论了特殊的行列式。'
                  '详细的知识点笔记：1. 二阶行列式的定义与计算 (重要程度：5)：（00:01）二阶行列式的定义：一个方程组的系数矩阵的主对角线元素之积，即对角线元素的乘积；（03：52）二阶行列式的计算方法：利用二阶行列式的定义，将方程组的系数矩阵化为上三角矩阵，再利用对角线元素的乘积求和。2. 三阶行列式的定义与计算 (重要程度：5)：（05：44）三阶行列式的定义：一个方程组的系数矩阵的主对角线元素之积，即对角线元素的乘积；（08：36）三阶行列式的计算方法：利用代数余子式的性质，利用三阶行列式的定义，将方程组的系数矩阵化为上三角矩阵，再利用对角线元素的乘积求和。3.特殊行列式（重要程度：3）：（15:03)上三角行列式：是主对角线以下的元素均为零的行列式，其值等于主对角线元素的乘积;（16:27)下三角行列式：主对角线以上的元素均为零的行列式，其值等于主对角线元素的乘积。关键词及参考网站：行列式 - 维基百科：https://zh.wikipedia.org/wiki/%E8%A1%8C%E5%88%97%E5%BC%8F；线性代数 - 可汗学院：https://zh.khanacademy.org/math/linear-algebra'
                  ),
        ('human','{record_text}')
    ]
)
time_model1_build = time.time()
print("模型1构建成功，当前步骤耗时：", f"{time_model1_build-time0:.2f}s",f"总耗时：{time_model1_build-start_time:.2f}s")

chain_str = prompt_template_str|model_str
note_text_all = chain_str.invoke(input = {'record_text': record_text})

model1_name,model1_finishReason=note_text_all.response_metadata['model_name'],note_text_all.response_metadata['finish_reason']
note_text = note_text_all.content
pprint(note_text_all)
time1 = time.time()
print("模型1运行成功，当前步骤耗时：", f"{time1-time_model1_build:.2f}s",f"总耗时：{time1-start_time:.2f}s")
# print(note_text)
print("模型1输出token数：",len(note_text))

if model1_finishReason=='length':
    model_str = ChatOpenAI(
        model="glm-4-long",
        openai_api_base=base_url,
        max_tokens=4095,
        # response_format={'type':'json_object'}
    )
    chain_str = prompt_template_str | model_str
    note_text_all = chain_str.invoke(input={'record_text': record_text})
    model1_name, model1_finishReason = note_text_all.response_metadata['model_name'], note_text_all.response_metadata[
        'finish_reason']
    note_text = note_text_all.content
    time_model1_retry = time.time()
    print("token超限，模型1(长文本)运行成功，当前步骤耗时：",f"{ time_model1_retry-time1:.2f}s",f"总耗时：{time_model1_retry-start_time:.2f}s")
    pprint(note_text_all)



if len(note_text) > 4000:#纯摆设 符号很多的数学类待考虑 口语化的课程1000就会超限
    model2_name='glm-4-long'
else:
    model2_name='glm-4-air-0111'

model_json = ChatOpenAI(
    model=model2_name,
    openai_api_base=base_url,
    max_tokens=4095,
    # response_format={'type':'json_object'}
)

model_json_massage = ("接下来把用户输入的课堂笔记进行拆分并输出一个.json文件代码(不是让你写一个用于转换成json的程序，而是直接输出json文件)，不分行，除了代码什么都不要，包含：1.课程主题。2.详细的知识点笔记(知识点是由字典组成的列表，每个知识点包含名称、重要程度（整数1-5，5代表最重要）、子要点（每一个知识点要有对应的解释或例子，由子要点组织，即子要点是知识点内容的具体阐述、该知识点开始时间以及这个知识点对应的全部文本，子要点的名字要是一个简洁的词语或短语）；子要点也是由字典组成的列表，键和值用markdown格式，键用一级标题，值可以适当加粗或斜体等）、以字典列表组织的每个子要点对应的开始时间。3.关键词及对应的参考网站。4.课程摘要。示例如下："
                    #"{{'theme'：'二阶和三阶行列式的定义及其计算方法', 'points'：[{{'name：'#二阶行列式','importance':4, 'subtitles'：[{{'#定义':'二阶行列式由两个正项相乘减去两个数负项相乘得到。'}}, {{'#应用'：'解二元一次方程'}}]}},[{{'name：'#特殊行列式','importance':2, 'subtitles'：[{{'#上三角行列式':'所有主对角以下的元素**为零**的行列式。'}}, {{'#下三角行列式'：'有主对角以上的元素**为零**的行列式'}}]}}],"
                    "{{'theme'：'二阶和三阶行列式的定义及其计算方法', 'points'：[{{'name：'#二阶行列式','importance':4, 'subtitles'：[{{'subtitle':'#定义','md':'**二阶行列式**由两个正项相乘减去两个数负项相乘得到。','raw_recognition':[{{'start':00:00,'end':01:58}}]}},"
                      # " {{'subtitle':'#应用','md':'解二元一次方程。',raw_recognition':[{{'start':01:58,'end':03:00]}}]}}]}}],[{{'name：'#特殊行列式','importance':2, 'subtitles'：[{{'subtitle':'#上三角行列式','md':{{'所有**主对角以下**的元素**为零**的行列式。'}},raw_recognition':[{{'start':03:00,'end':04:00,''text':'我们看这个题，而且我们来呗。啊。第一个这么想，第二个政策呢？咱甭写了，因为它有零。第三个正向呢？对吧？也有零吗？第一个负向是不是有零啊？第二个负向，这没有礼貌。第三个附向也有理吗？也有理吗？那就这么着了呗，对吧？那这个行列式啊，我们其实是有一个特殊的名字，哈？它其实是叫一个上三角。行列式。'}}],'"
                      "{{'subtitle':'#应用','md':'解二元一次方程。',raw_recognition':[{{'start':01:58,'end':03:00]}}]}}]}}],"
                      "[{{'name：'#特殊行列式','importance':2, 'subtitles'：[{{'subtitle':'#上三角行列式','md':{{'所有**主对角以下**的元素**为零**的行列式。'}},raw_recognition':[{{'start':03:00,'end':04:00}}]}},'"
                      # "{{'subtitle':'#下三角行列式','md':'所有主对角以上的元素**为零**的行列式'，raw_recognition':[{{'start':04:00,'end':05:00,'text':'我们后边还会讲n结的。哈？那么，它的这个定义也非常简单。就是，你看，这不是上面有东西吗？下边就是没东西嘛，下边全都零嘛。这种就叫做上三段行业是它的结果啊。就等于主标线员不想省就完事了。那再来一个这个呢？叫做一个下三角行列式啊。下半年好像是它的主要呢。'}}]}}],"
                      "{{'subtitle':'#下三角行列式','md':'所有主对角以上的元素**为零**的行列式'，raw_recognition':[{{'start':04:00,'end':05:00]}}]}}]}}],"
                    "'links':[{{'name'：'MIT OpenCourseWare'，'href'：'https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010'}}, {{'name'：'Wolfram MathWorld'，'href'：'hhttps://mathworld.wolfram.com/Determinant.html'}}],"
                      "'summary'：'二阶行列式由两个正项相乘减去两个数负项相乘得到。三阶行列式由九个元素组成，计算时需要考虑三个正项和三个负项。正项是沿着主对角线、副对角线以及从左上角到右下角的对角线上的元素乘积。负项是沿着主对角线、副对角线以及从右上角到左下角的对角线上的元素乘积。三阶行列式的计算使用对角线法则，将三阶行列式展开为三个正项和三个负项的和。特殊行列式包括上三角行列式、下三角行列式和对角行列式。'}})")

prompt_template_json = ChatPromptTemplate.from_messages(
    [
        ('system',model_json_massage),
        ('human',note_text)
    ]
)
time_model2_build = time.time()
print("模型2构建成功，当前步骤耗时：", f"{time_model2_build-time1:.2f}s",f"总耗时：{time_model2_build-start_time:.2f}s")

chain_json = prompt_template_json|model_json
json_text_all = chain_json.invoke({})
json_text = json_text_all.content
time2 = time.time()
print("模型2运行成功，当前步骤耗时：",f"{ time2-time_model2_build:.2f}s",f"总耗时：{time2-start_time:.2f}s")
model2_name,model2_finishReason=json_text_all.response_metadata['model_name'],json_text_all.response_metadata['finish_reason']
pprint(json_text_all)

if model2_finishReason=='length':
    model_json = ChatOpenAI(
        model="glm-4-long",
        openai_api_base=base_url,
        max_tokens=4095,
        # response_format={'type':'json_object'}
    )
    prompt_template_json = ChatPromptTemplate.from_messages(
        [
            ('system', model_json_massage),
            ('human', note_text)
        ]
    )
    chain_json = prompt_template_json | model_json
    json_text_all = chain_json.invoke({})
    json_text = json_text_all.content
    time3 = time.time()
    print("token超限，模型2(长文本)运行成功，当前步骤耗时：",f"{ time3-time2:.2f}s",f"总耗时：{time3-start_time:.2f}s")
    pprint(json_text_all)
    model2_name, model2_finishReason = json_text_all.response_metadata['model_name'], json_text_all.response_metadata[
        'finish_reason']

note_text = note_text.replace('{', '{{').replace('}', '}}')

# 输出文件
output_file1 = f'files/{file_name_nopostfix}_{time_now}_{model1_name}.md'
output_file2 = f'files/{file_name_nopostfix}_{time_now}_{model2_name}.json'

with open(output_file1, 'w', encoding='utf-8') as f:
    f.write(note_text)

json_text = json_text.replace('```json','').replace('```','')

with open(output_file2, 'w', encoding='utf-8') as f:
    f.write(json_text)

time3 = time.time()
print("输出文件成功，",f"总耗时：{time3-start_time:.2f}s")

