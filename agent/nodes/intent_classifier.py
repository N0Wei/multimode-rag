#!/usr/bin/env python
# coding: utf-8
"""
意图分类节点
识别用户问题类型
"""

from typing import Dict
from agent.state import AgentState, format_conversation_history
from agent.config.model_config import ModelConfig


def intent_classify_node(state: AgentState) -> AgentState:
    """
    意图分类节点
    
    意图类型:
    - product_inquiry: 产品咨询 (需要 RAG)
    - complaint: 投诉 (需要转人工)
    - faq: 常见问题 (可能有标准答案)
    - general: 一般性问题
    - chitchat: 闲聊
    """
    question = state["user_question"]
    history = state.get("conversation_history", [])
    
    # 使用 LLM 进行意图分类
    intent = _classify_with_llm(question, history)
    
    state["intent"] = intent
    state["metadata"] = state.get("metadata", {})
    state["metadata"]["intent_classification"] = intent
    
    print(f"  [Intent] 意图识别: {intent}")
    
    return state


def _classify_with_llm(question: str, history: list) -> str:
    """
    使用 LLM 进行意图分类
    
    Args:
        question: 用户问题
        history: 对话历史
        
    Returns:
        意图类型: product_inquiry | complaint | faq | general | chitchat
    """
    try:
        llm = ModelConfig.get_intent_llm()
        
        # 构建提示词
        history_text = ""
        if history:
            history_text = f"\n\n对话历史:\n{format_conversation_history(history, max_turns=3)}"
        
        prompt = f"""你是一个智能客服系统的意图识别助手。请分析用户问题，判断属于以下哪一类意图：

**意图类型定义**:

1. **product_inquiry** (产品咨询)
   - 询问产品参数、配置、功能、性能
   - 例如: "银河E8的续航是多少？"、"这款车有什么配置？"、"支持快充吗？"

2. **complaint** (投诉/不满)
   - 表达不满、投诉、要求退款/赔偿
   - 例如: "要投诉你们的服务"、"质量太差了要退款"、"客服态度恶劣"

3. **faq** (常见问题)
   - 关于售后、保修、客服联系方式等通用问题
   - 例如: "如何联系客服？"、"保修期是多久？"、"营业时间是几点？"

4. **general** (一般性咨询)
   - 不属于上述类别的正常咨询
   - 例如: "有什么优惠活动？"、"怎么预约试驾？"

5. **chitchat** (闲聊/寒暄)
   - 打招呼、感谢、告别、闲聊
   - 例如: "你好"、"谢谢"、"再见"、"今天天气怎么样？"

**重要规则**:
- 如果问题明确提到具体产品(如"银河E8")的技术参数，选择 **product_inquiry**
- 如果包含负面情绪词(投诉、不满、退款等)，选择 **complaint**
- 如果询问通用服务(客服电话、保修政策等)，选择 **faq**
- 在多轮对话中，结合上下文判断意图{history_text}

**用户问题**: {question}

请直接输出意图类型，只输出以下五个选项之一，请不要输出额外的内容，这会干扰我后续的字符匹配工作：
product_inquiry
complaint
faq
general
chitchat

意图类型:"""

        response = llm.invoke(prompt)
        intent_raw = response.content.strip().lower()
        
        # 提取意图（处理可能的多余文本）
        valid_intents = ["product_inquiry", "complaint", "faq", "general", "chitchat"]
        
        for valid_intent in valid_intents:
            if valid_intent in intent_raw:
                print(f"  [LLM 分类] {intent_raw} → {valid_intent}")
                return valid_intent
        
        # 如果 LLM 输出不合法，降级到规则分类
        print(f"  [警告] LLM 输出无效: {intent_raw}, 降级到规则分类")
        return _classify_by_rules(question)
        
    except Exception as e:
        print(f"  [警告] LLM 意图分类失败: {e}, 降级到规则分类")
        return _classify_by_rules(question)


def _classify_by_rules(question: str) -> str:
    """
    基于规则的意图分类（作为 LLM 失败时的降级方案）
    """
    question_lower = question.lower()
    
    # 投诉关键词
    complaint_keywords = ["投诉", "不满意", "质量问题", "退款", "赔偿", "客服态度", "太差", "恶劣"]
    if any(kw in question for kw in complaint_keywords):
        return "complaint"
    
    # 闲聊关键词
    chitchat_keywords = ["你好", "谢谢", "再见", "天气", "笑话", "早上好", "晚安"]
    if any(kw in question for kw in chitchat_keywords):
        return "chitchat"
    
    # FAQ 关键词
    faq_keywords = ["客服", "电话", "联系方式", "营业时间", "保修", "售后", "怎么联系"]
    if any(kw in question for kw in faq_keywords):
        return "faq"
    
    # 产品咨询 (包含产品名称或技术词汇)
    product_keywords = ["银河", "吉利", "E8", "L7", "续航", "充电", "快充", "配置", "参数", "功能", "性能", "马力", "扭矩"]
    if any(kw in question for kw in product_keywords):
        return "product_inquiry"
    
    # 默认为一般性问题
    return "general"
