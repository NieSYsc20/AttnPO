
import re, time
from tqdm import tqdm
import numpy as np
import os
from typing import List, Dict, Any, Optional
import json
from collections import defaultdict
import math

# 1. 定义词表
zhuanzhe = [  
        "Wait",
        "But",  # "But wait"
        "However",
        "Hold on",
        "Looking back",
        "I don't see any errors",
        "Hmm, wait",
        "Hmm, no",
        "Hmm, but",
        "Hmm, just",
        "Hmm, let me double-check",
        "Hmm, let me verify",
        "Hmm, let me make sure", 
        "Hmm, let me check", 
        "No,"
        # double-check
        "Let me double-check",
        "Let me just double-check",
        "Let's double-check",
        "Let's just double-check",
        "Just to double-check",
        # verify
        "Let me verify",
        "Let me just verify",
        "Let's verify",
        "Let's just verify",
        "Just to verify",
        # confirm
        "Let me confirm",
        "Let me just confirm",
        "Let's confirm",
        "Let's just confirm",
        "Just to confirm",
        # check
        "Let me check",
        "Let me just check",
        "Let's check",
        "Let's just check",
        "Just to check",
        # recap
        "Let me recap",
        "Let me just recap",
        "Let's recap",
        "Let's just recap",
        "Just to recap",
        # make sure
        "Let me make sure",
        "Let me just make sure",
        "Let's make sure",
        "Just to make sure",
        # 表达疑惑的语气
        "So, is",
        "Is it",
        "Is that right?",
        "Is there",
        "Right?",
    ]

fencha = [  
        "Alternatively",
        "Another approach",
        "Another way",  
        "Another idea", 
        "Another thought", 
        "I guess another way",
        "I guess another approach",
        "Let me just think if there",
    ]


backward = [    # 总结前述推理
        "So",
        "Therefore",
        "Thus",
        "Hence",
        "Hmm, so",
        "Okay, so",
        "Putting it all together",
    ]


forward = [     # 开启后续推理
        "Let me",
        "Let's",
        "I ", # 注意此处必须有空格 防止匹配其他单词的前缀"
        "We ",  # 添加
        
        "Okay",
        "Hmm",
        "Now",
        "Alright",

        "First,",
        "First off,",
        "Second,",
        "Third,",
        "Starting",
        "Then",
        "Next",
        "Finally",
        "Similarly",
        "Again",
        
        "In this case",
        "Because",
        
        "Given that",
        "The problem gives",

        "To ",
        "Since",
    ]
special = [
        "**Final Answer",
    ]

all_words = zhuanzhe + fencha + forward + backward + special
confuse_words = zhuanzhe + fencha
tuijin_word = forward + backward


# 2. 构造正则
sorted_words = sorted(all_words, key=len, reverse=True)
pattern = r'(' + '|'.join(map(re.escape, sorted_words)) + r')'
all_words_regex = re.compile(pattern)

def segment_cot(rollout, min_segment_chars: int = 80, min_onther_chars: int = 15):
    """
    如果没有思考过程的话，返回的是空列表
    """
    rollout = rollout.split("</think>")[0]
    if len(rollout) == 0:
        return []
    else:
        # 3. 进行粗匹配。记录匹配到的 (start, word)（已按出现顺序）
        matches = []
        for m in all_words_regex.finditer(rollout):
            w = m.group(0)
            s = m.start()
            if w in tuijin_word:
                prefix = rollout[max(0, s - 1): s]
                if prefix != "\n":  # 修改只有一个\n就可以分段了
                    continue
            matches.append((s, w))

        # positions:  [0] + 每个匹配起点 + [len]
        positions = [s for s, _ in matches]
        start2word = {s: w for s, w in matches}
    
        if 0 not in positions:
            positions = [0] + positions
            start2word[0] = None
        if len(rollout) not in positions:
            positions = positions + [len(rollout)]
            start2word[len(rollout)] = None

        def classify_trigger(w):
            if w in confuse_words:  # 1表示混淆词
                return 1
            if w in forward:  # 2表示向前合并
                return 2
            if w in backward:  # 3表示向后合并
                return 3
            if w in special:  # 4表示特殊处理
                return 4
            if w is None: # 第一句默认为向后合并
                return 2


        segments: List[Dict[str, Any]] = []
        for i in range(len(positions) - 1):
            seg_start, seg_end = positions[i], positions[i + 1]
            seg_text = rollout[seg_start:seg_end]
            trigger = start2word.get(seg_start)
            kind = classify_trigger(trigger)
            segments.append({
                "text": seg_text,
                "trigger": trigger,
                "kind": kind,
                "start": seg_start,
                "end": seg_end,
            })

        # 判断某个片段是否过短
        def too_short(s: str) -> bool:
            return len(s["text"]) < min_segment_chars

        # 将 idx 段与右侧段合并到 idx 位置，成功返回 True。
        def merge_right(idx: int):
            a, b = segments[idx], segments[idx + 1]
            merged = {
                "text": a["text"] + b["text"],
                "trigger": a["trigger"],      # 合并后段起点仍为左段起点
                "kind": a["kind"],
                "start": a["start"],
                "end": b["end"],
            }
            segments[idx:idx + 2] = [merged]


        # 将 idx 段并入左侧段（结果落在 idx-1 位置），成功返回 True。
        def merge_left(idx: int):
            left, right = segments[idx - 1], segments[idx]
            merged = {
                "text": left["text"] + right["text"],
                "trigger": left["trigger"],   # 起点保留为左段
                "kind": left["kind"],
                "start": left["start"],
                "end": right["end"],
            }
            segments[idx - 1:idx + 1] = [merged]
        
        def only_seg(seg) -> bool:
            text = seg["text"]
            trig = seg["trigger"] or ""
            return len(text) - len(trig) <= min_onther_chars

        # 合并 final_answer永远单独算一步，防止凭空增加其前面步骤的得分。一个是final_answer不合并，另外一个是向右边合并时，如果遇到final_answer就需要不合并
        i = 0
        while i < len(segments):
            seg = segments[i]
            if too_short(seg) and seg["kind"] != 4:
                if seg["kind"] == 1:   # 如果是困惑步骤且其右侧步骤不是**Final Answer，那么就向右合并，否则不处理不合并
                    if i < len(segments) - 1 and segments[i+1]["kind"] != 4: # 如果可以进行右侧合并
                        merge_right(i)  # 进行右侧合并
                        continue
                elif seg["kind"] == 2 or seg["kind"] == 3:   # 如果是铺垫推进或者收束推进
                    if i + 1 < len(segments) - 1 and segments[i+1]["kind"] == 1:  # 如果下一个步骤是困惑步骤，就向左合并
                        if i > 0: # 如果可以向左合并
                            merge_left(i) # 向左合并
                            i = i - 1
                            continue 
                    elif seg["kind"] == 2:  # 推进步骤为铺垫推进，且其下一个步骤不为困惑步骤和特殊步骤
                        if i < len(segments) - 1 and segments[i+1]["kind"] != 4:   # 如果可以进行右侧合并
                            merge_right(i)  # 进行右侧合并
                            continue
                    elif seg["kind"] == 3:  # 推进步骤为收束推进，且其下一个步骤不为困惑步骤
                        if only_seg(seg) and i < len(segments)-1 and segments[i+1]["kind"] != 4:  # 如果收束步骤仅仅一句话，且可以向右合并
                            merge_right(i)
                            continue
                        if i > 0: # 如果可以进行左侧合并
                            merge_left(i)  # 向左合并
                            i = i - 1
                            continue
            i += 1  # continue表示一旦发生合并，不再执行该循环内后续代码（不进行i+1）,而是重新判断该步骤的长度是否太短；仅有不满足too_short(seg)才进行i+1
            
        return segments

# rollout = """Okay, so Wendi has 20 chickens, right? Every day, she feeds each chicken three cups of mixed feed. That means for one chicken, it's 3 cups. So, for 20 chickens, that should be 20 times 3 cups, which is 60 cups per day. Hmm, wait, no, hold on. Let me read the problem again.\n\nIt says, \"Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?\"\n\nWait, so maybe the initial statement about three cups per chicken per day is a general statement, and then the problem gives specific numbers for morning and afternoon feedings. So, perhaps the daily feed is 3 cups per chicken, but the problem is giving specific numbers for the morning and afternoon, and we need to find the remaining feed needed for the final meal.\n\nSo, Wendi has 20 chickens. In the morning, she gives 15 cups. In the afternoon, 25 cups. So, total feed given so far is 15 + 25 = 40 cups. Since each chicken needs 3 cups a day, for 20 chickens, that's 60 cups per day. So, the total feed needed for the day is 60 cups. She has already given 40 cups, so the remaining is 60 - 40 = 20 cups. Therefore, she needs to give 20 cups in the final meal.\n\nWait, that seems straightforward. Let me check:\n\nTotal daily feed: 20 chickens * 3 cups = 60 cups.\n\nGiven in morning: 15 cups.\n\nGiven in afternoon: 25 cups.\n\nTotal given so far: 15 + 25 = 40 cups.\n\nRemaining feed: 60 - 40 = 20 cups.\n\nYes, that makes sense.\n\n**Final Answer**\nThe number of cups Wendi needs to give in the final meal is \\boxed{20}.\n</think>\n\nWendi has 20 chickens, and each chicken is fed 3 cups of mixed feed daily. Therefore, the total feed needed per day is:\n\n\\[ 20 \\text{ chickens} \\times 3 \\text{ cups/chicken} = 60 \\text{ cups} \\]\n\nWendi gives 15 cups in the morning and 25 cups in the afternoon. The total feed given so far is:\n\n\\[ 15 \\text{ cups} + 25 \\text{ cups} = 40 \\text{ cups} \\]\n\nThe remaining feed needed for the final meal is:\n\n\\[ 60 \\text{ cups} - 40 \\text{ cups} = 20 \\text{ cups} \\]\n\nThus, the number of cups Wendi needs to give in the final meal is \\boxed{20}."""

# a = segment_cot(rollout)
# for i in a:
#     print(i["text"])
#     print("="*40)