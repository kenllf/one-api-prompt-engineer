LARGE_MODEL = "claude-3-opus-20240229" # The large model to use

# SMALL_MODEL = "claude-3-haiku-20240307" # The small model to use
SMALL_MODEL = "glm-4"

import openai
import re

openai.api_key = "sk-xpA4UDDjmVQU8oRp0cA2EaC9111f401483A2E9E957C02dBb"
openai.base_url = "http://127.0.0.1:3000/v1/"

def generate_candidate_prompts(task, prompt_example, response_example):
    messages = [{
            "role": "system",
            "content":"""<task>Given an example training sample, create seven additional samples for the same task that are even better. Each example should contain a <prompt> and a <response>.</task>

<rules>
1. Ensure the new examples are diverse and unique from one another.
2. They should all be perfect. If you make a mistake, this system won't work.
</rules>

Respond in this format:
<response_format>
<example_one>
<prompt>
PUT_PROMPT_HERE
</prompt>
<response>
PUT_RESPONSE_HERE
</response>
</example_one>

<example_two>
<prompt>
PUT_PROMPT_HERE
</prompt>
<response>
PUT_RESPONSE_HERE
</response>
</example_two>

...
</response_format>"""
        }, {
            "role": "user",
            "content": f"""<training_task>{task}</training_task>

<prompt_example>
{prompt_example}
</prompt_example>

<response_example>
{response_example}
</response_example>"""},
    ]

    response = openai.chat.completions.create(
        model=LARGE_MODEL,
        max_tokens=4000,
        temperature=0.5,
        messages=messages,
    )
    response_text = response.choices[0].message.content

    # Parse out the prompts and responses
    prompts_and_responses = []
    examples = re.findall(r'<example_\w+>(.*?)</example_\w+>', response_text, re.DOTALL)
    for example in examples:
        prompt = re.findall(r'<prompt>(.*?)</prompt>', example, re.DOTALL)[0].strip()
        response = re.findall(r'<response>(.*?)</response>', example, re.DOTALL)[0].strip()
        prompts_and_responses.append({'prompt': prompt, 'response': response})

    return prompts_and_responses

def generate_system_prompt(task, prompt_examples):
    messages = [
        {"role": "system", "content": """<your_role>Given a user-description of their <task> a set of prompt / response pairs (it'll be in JSON for easy reading) for the types of outputs we want to generate given inputs, write a fantastic system prompt that describes the task to be done perfectly.</your_role>

<rules>
1. Do this perfectly.
2. Respond only with the system prompt, and nothing else. No other text will be allowed.
</rules>

Respond in this format:
<system_prompt>
WRITE_SYSTEM_PROMPT_HERE
</system_prompt>"""
        },
        {"role": "user", "content": f"""<task>{task}</task>

<prompt_response_examples>
{str(prompt_examples)}
</prompt_response_examples>"""
        }]

    response = openai.chat.completions.create(
        model=LARGE_MODEL,
        max_tokens=1000,
        temperature=0.5,
        messages=messages,
    )
    response_text = response.choices[0].message.content

    # Parse out the prompt
    system_prompt = response_text.split('<system_prompt>')[1].split('</system_prompt>')[0].strip()

    return system_prompt

def test_haiku(generated_examples, prompt_example, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]

    for example in generated_examples:
      messages.append({"role": "user", "content": example['prompt']})
      messages.append({"role": "assistant", "content": example['response']})

    messages.append({"role": "user", "content": prompt_example.strip()})

    response = openai.chat.completions.create(
        model=SMALL_MODEL,
        max_tokens=2000,
        temperature=0.5,
        messages=messages,
    )
    response_text = response.choices[0].message.content

    return response_text

def run_haiku_conversion_process(task, prompt_example, response_example):

    print('Generating the prompts / responses...')
    # Generate candidate prompts
    generated_examples = generate_candidate_prompts(task, prompt_example, response_example)

    print('Prompts / responses generated. Now generating system prompt...')

    # Generate the system prompt
    system_prompt = generate_system_prompt(task, generated_examples)

    print('System prompt generated:', system_prompt)


    print('\n\nTesting the new prompt on '+SMALL_MODEL+', using your input example...')
    # Test the generated examples and system prompt with the Haiku model
    small_model_response = test_haiku(generated_examples, prompt_example, system_prompt)

    print(SMALL_MODEL+' responded with:')
    print(small_model_response)

    print('\n\n!! CHECK THE FILE DIRECTORY, THE PROMPT IS NOW SAVED THERE !!')

    # Create a dictionary with all the relevant information
    result = {
        "task": task,
        "initial_prompt_example": prompt_example,
        "initial_response_example": response_example,
        "generated_examples": generated_examples,
        "system_prompt": system_prompt,
        "small_model_response": small_model_response
    }

    # Save the Haiku prompt to a Python file
    with open("haiku_prompt.py", "w") as file:
        file.write('system_prompt = """' + system_prompt + '"""\n\n')

        file.write('messages = [\n')
        for example in generated_examples:
            file.write('    {"role": "user", "content": """' + example['prompt'] + '"""},\n')
            file.write('    {"role": "assistant", "content": """' + example['response'] + '"""},\n')

        file.write('    {"role": "user", "content": """' + prompt_example.strip() + '"""}\n')
        file.write(']\n')

    return result



# task = "refactoring complex code"
task = "根据提供信息写一份旅游指南"

prompt_example = """这份靠谱的西安游玩攻略，你能想象是一位理工科的男友写的吗🤣真的太不可思议了，哈哈哈哈！！！各位姐妹们可以认真看完哦🥳
-
💦西安行程安排
【📍第①天】钟楼＞鼓楼＞西安城墙＞回民街＞高家大院（看皮影戏哦）
【📍第②天】陕西历史博物馆（周一闭馆，需预约）＞洒金桥＞大雁塔（可以不用登塔）＞大唐不夜城＞大唐芙蓉园
【📍第③天】华清池＞骊山＞秦始皇兵马俑（建议请讲解，早点去）＞丽山园＞长恨歌（很震撼的表演）"""
# prompt_example = """def calculate_total(prices, tax, discount, shipping_fee, gift_wrap_fee, membership_discount):

#     total = 0

#     for i in range(len(prices)):

#         total += prices[i]

#     if membership_discount != 0:

#         total = total - (total * (membership_discount / 100))

#     if discount != 0:

#         total = total - (total * (discount / 100))

#     total = total + (total * (tax / 100))

#     if total < 50:

#         total += shipping_fee

#     else:

#         total += shipping_fee / 2

#     if gift_wrap_fee != 0:

#         total += gift_wrap_fee * len(prices)

#     if total > 1000:

#         total -= 50

#     elif total > 500:

#         total -= 25

#     total = round(total, 2)

#     if total < 0:

#         total = 0

#     return total"""

response_example = """西安，作为⼗三朝古都
我前前后后去了5次
真的是好吃，好拍，好逛的城市❤️
这次又在西安暴走3天
-
整理了1⃣️8⃣️个超出片的机位，跟着走就对了‼️
-
1⃣️大慈恩寺遗址公园
免费公园，可以get佛像和大雁塔同框
2⃣️西安城墙（永宁门）
导航“永宁门”，西侧城墙，白天&夜景都好看
3⃣️开元商场五楼（鼓楼店）
这里可以俯瞰鼓楼，夜景绝美
4⃣️永宁门
售票处上楼正对着可以拍到鼓楼
5⃣️广仁寺
沿着左侧的红墙走就是出片机位
6⃣️青龙寺
相对比较冷门的地方，但是很有大唐盛世的韵味。
7⃣️大悦城顶楼
拍大雁塔的好机位，晚上大雁塔会亮灯
8⃣️子佩集
在回民街，早上9点开门就去，不然人会多
9⃣️湘子庙街
有各种好看的店铺，这家真的很显眼出片。
1⃣️0⃣️鼓楼
站在南侧路对面可以拍到完整的古楼
1⃣️1⃣️大雁塔南广场
顺时针走一圈，有不少跟大雁塔同框的机位
1⃣️2⃣️贰回巷火锅馆（钟楼店）
门口有“长安”字标
1⃣️3⃣️辣斗辣火锅（钟楼店）
门口有“西安”字标
1⃣️4⃣️陆离汉服
门口有“长安”字标
1⃣️5⃣️大雁塔南广场（南门口）
1⃣️6⃣️永宁门（靠近出口的位置）
1⃣️7⃣️永宁门（靠近出口的位置）
1⃣️8⃣️永宁门（西安城墙夜景）
-
🚶「我的3天2晚游玩路线」
Day1:大雁塔—唐大慈恩寺—大悦城—钟楼
Day2:广仁寺—书院门—永宁门城墙
Day3:回民街—青龙寺—湘子庙街
-
🍜「美食攻略」
胡辣汤，羊肉泡馍这种针对游客的试试就行，本地人常去的洒金桥和湘子庙街，洒金桥小吃多，接地气，湘子庙街咖啡店多更时髦现代，适合打卡。
-
跟着这份详细攻略走，不出错。如果不想人多的，也可以避开节假日，错峰出行，拍照体验跟更好。"""
# response_example = """def calculate_total(prices, tax_rate, discount_rate, shipping_fee, gift_wrap_fee, membership_discount_rate):

#     def apply_percentage_discount(amount, percentage):

#         return amount * (1 - percentage / 100)

#     def calculate_shipping_fee(total):

#         return shipping_fee if total < 50 else shipping_fee / 2

#     def apply_tier_discount(total):

#         if total > 1000:

#             return total - 50

#         elif total > 500:

#             return total - 25

#         return total

#     subtotal = sum(prices)

#     subtotal = apply_percentage_discount(subtotal, membership_discount_rate)

#     subtotal = apply_percentage_discount(subtotal, discount_rate)



#     total = subtotal * (1 + tax_rate / 100)

#     total += calculate_shipping_fee(total)

#     total += gift_wrap_fee * len(prices)



#     total = apply_tier_discount(total)

#     total = max(0, round(total, 2))



#     return total"""

result = run_haiku_conversion_process(task, prompt_example, response_example)