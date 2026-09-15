import type { KeyStepKind } from "@/types";

export const STEP_LABELS: Record<string, string> = {
  Clarification: "澄清问题",
};

export const TOOL_LABELS: Record<string, string> = {
  traffic_analyze: "流量分析",
  ads_analyze: "广告分析",
  inventory_check: "库存检查",
  product_diagnose: "商品诊断",
  retrieve_knowledge: "知识检索",
  update_context: "记忆更新",
};

export const FIELD_LABELS: Record<string, string> = {
  latency_ms: "延迟(毫秒)",
  fallback_used: "是否使用兜底",
  ttfb_ms: "首包延迟(毫秒)",
  event_count: "事件数",
  event_completeness: "事件完整性",
  llm_latency_ms: "LLM耗时(毫秒)",
  tool_latency_ms: "工具耗时(毫秒)",
  loop_count: "循环轮数",
  retrieve_hits: "检索命中数",
  duration_ms: "耗时(毫秒)",
  thought: "思考",
  action: "动作",
  action_input: "动作输入",
  observation: "观察结果",
  result: "最终结果",
  query: "问题",
  context: "上下文",
  session_id: "会话ID",
  missing_context_keys: "缺失字段",
};

export const KEY_STEP_KIND_LABELS: Record<KeyStepKind, string> = {
  first_evidence: "证据首发",
  context_update: "记忆更新",
  direction_repair: "方向修正",
  context_update_rejected: "记忆更新被拒",
};

export const CONSTRAINT_STATUS_LABELS: Record<string, string> = {
  satisfied: "已满足",
  unsatisfied: "未满足",
  unchecked: "未核查",
};

export const JOB_STATUS_LABELS: Record<string, string> = {
  queued: "排队中",
  running: "运行中",
  cancel_requested: "取消请求中",
  succeeded: "成功",
  degraded: "降级完成",
  failed: "失败",
  cancelled: "已取消",
};

export function toolLabel(name: string) {
  return TOOL_LABELS[name] ? `${TOOL_LABELS[name]} (${name})` : name;
}

export function fieldLabel(key: string) {
  return FIELD_LABELS[key] ?? key;
}