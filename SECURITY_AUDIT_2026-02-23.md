# 全量代码审计报告（v5-trading-bot）

审计时间：2026-02-23  
审计范围：`main.py`、`src/`、`configs/`、`scripts/`、依赖清单 `requirements.txt`

## 审计方法
- 手工代码走查（重点关注：路径拼接、凭据处理、危险操作闸门、异常吞噬、供应链风险）。
- 静态模式检索（`rg`）检查高风险 API：`eval/exec`、`subprocess`、`yaml.load`、`pickle`、`shell=True` 等。
- 运行基础测试命令与依赖漏洞扫描可行性检查。

---

## 结论摘要
- **发现 1 个中危漏洞（建议尽快修复）**：`run_id` 未做约束直接拼接路径，存在目录穿越/任意路径写入风险。
- **发现 2 个低危安全问题**：
  1. `.env` 默认覆盖系统环境变量，存在配置投毒风险；
  2. 实盘高危脚本仅依赖单一环境变量闸门，缺少强认证/二次确认，存在误触发与环境污染后被滥用风险。
- **发现 1 个供应链治理问题（低危）**：依赖版本未锁定（且无 hash pin），不利于防范依赖投毒与可重现构建。

---

## 详细发现

### F-01: `run_id` 可控导致路径穿越与越界写入（中危）
**风险等级**：Medium  
**影响面**：运行时产物目录、日志目录、报告目录（可能被写到项目目录外）

#### 证据
- `run_id` 直接来自环境变量：`run_id = os.getenv("V5_RUN_ID")`（`main.py`）。
- 该值随后直接拼接到路径，如：
  - `RunLogger(run_dir=f"reports/runs/{run_id}")`
  - `Path(f"reports/runs/{run_id}/spread_snapshot.json").write_text(...)`
  - `audit.save(f"reports/runs/{run_id}")`
- `RunLogger` 会 `mkdir(parents=True, exist_ok=True)` 并创建/追加文件，未做路径归一化与基目录约束（`src/core/run_logger.py`）。

#### 攻击/误用场景
若运行环境（CI、systemd、cron、运维脚本）可被注入环境变量，攻击者可设置：
`V5_RUN_ID=../../../../tmp/pwn`，将运行产物写出 `reports/runs/` 之外。

#### 修复建议
1. 对 `run_id` 做白名单校验（仅允许 `^[A-Za-z0-9_.-]{1,64}$`）。
2. 统一使用 `safe_join(base, user_part)`：`resolved_path.is_relative_to(base.resolve())` 校验。
3. 所有 `reports/runs/{run_id}` 路径写入点集中封装，避免分散遗漏。

---

### F-02: `.env` 加载使用 `override=True`，存在配置投毒风险（低危）
**风险等级**：Low  
**影响面**：交易环境变量（API 凭据、模式开关、风控阈值）

#### 证据
- `configs/loader.py` 中 `load_dotenv(env_path, override=True)` 会让 `.env` 覆盖已有环境变量。

#### 风险说明
在共享主机/不严格文件权限场景中，如果 `.env` 被修改，可篡改关键开关（如实盘闸门变量名/值）或替换凭据。

#### 修复建议
1. 生产环境改为 `override=False`。
2. 将 `.env` 仅用于开发；生产改用只读 Secret 管理（Vault/KMS/CI Secret）。
3. 启动时记录并告警“关键变量来源”。

---

### F-03: 高危交易脚本闸门过于单一（低危）
**风险等级**：Low  
**影响面**：实盘交易误触发、自动化任务误执行

#### 证据
- 例如 `scripts/live_sell_all.py` 仅检查：`if os.getenv(arm_env) != arm_val: raise ...`，通过后即可下单。

#### 风险说明
当前依赖“单个环境变量值”作为最终授权，在自动化部署或环境污染情况下容易被绕过（尤其与 F-02 组合）。

#### 修复建议
1. 引入“双因子闸门”：环境变量 + 时间窗一次性 token（HMAC 签名）。
2. 对“全卖/紧急脚本”要求交互式二次确认（非 TTY 场景拒绝执行）。
3. 强制写审计日志（操作者、来源主机、git sha、审批单号）。

---

### F-04: 依赖未锁定版本/哈希（供应链低危）
**风险等级**：Low  
**影响面**：可重现构建、依赖投毒防护

#### 证据
- `requirements.txt` 使用 `>=` 范围约束，无精确版本与哈希锁定。

#### 风险说明
当上游包发布恶意/破坏性版本时，自动安装存在被动升级风险。

#### 修复建议
1. 使用 `pip-tools` 生成锁定文件（`requirements.lock`）。
2. 对生产构建启用 `--require-hashes`。
3. 在 CI 增加依赖漏洞扫描（`pip-audit`/`safety`）。

---

## 本次未发现
- 未发现 `eval/exec`、`shell=True`、`yaml.load(unsafe)`、`pickle` 反序列化等高危模式。
- 未发现明文硬编码 API 密钥（配置中为 `${ENV}` 占位符）。

## 建议整改优先级
1. **P1（本周）**：修复 F-01 路径穿越风险（统一路径安全封装）。
2. **P2（本周）**：生产环境关闭 `.env override`，完善密钥来源治理（F-02）。
3. **P3（两周内）**：升级实盘脚本闸门策略（F-03）。
4. **P3（两周内）**：依赖锁定与漏洞扫描入 CI（F-04）。

